# PTDR Training Scaffold

This directory adds a repo-local training workflow for:

- `DBNet++` text detection through `MMOCR`
- `PARSeq` text recognition through the official `strhub` modules

Both entry points rebuild manifests from the PTDR folders on each run, skip malformed annotations by default, and support Weights & Biases logging.
They also support mixing additional train-only datasets into DBNet and PARSeq while keeping PTDR as the sole source of validation and test data.
The YAML files are `pyrallis`-compatible, so you can override nested fields directly from the CLI.

## Files

- `validate_dataset.py`: reports malformed detection and recognition annotations.
- `build_detection_manifest.py`: converts PTDR detection labels into MMOCR JSON manifests.
- `build_recognition_manifest.py`: converts PTDR recognition crops, plus weighted train-only external datasets, into LMDB datasets for PARSeq.
- `external_datasets.py`: adapters, weighted train mixing, and crop generation for supported external datasets.
- `train_dbnetpp.py`: launches DBNet++ training with MMOCR.
- `train_parseq.py`: launches PARSeq training with PyTorch Lightning.
- `end_to_end_utils.py`: shared checkpoint loading, end-to-end inference, metrics, and visualization helpers.
- `evaluate_end_to_end.py`: evaluates DBNet + PARSeq together on PTDR scene images.
- `notebooks/end_to_end_inference.ipynb`: interactive walkthrough for end-to-end inference and visual debugging.
- `configs/dbnetpp.yaml`: default detection experiment config.
- `configs/parseq.yaml`: default recognition experiment config.

## Install

1. If you need CUDA wheels, install the pinned `torch` and `torchvision` versions from the official PyTorch index first.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

The requirements pin `setuptools<81` to keep the OpenMMLab and PyTorch packaging stack stable, and `wandb>=0.22.3` so current W&B API keys work without the legacy 40-character key limitation.

If `mmcv` fails to install from pip, install it with `openmim`:

```bash
mim install "mmcv<2.2.0"
```

## Validate

```bash
python experiments/ptdr/validate_dataset.py --include-domain outdoor_text --include-domain indoor_text
```

## External Datasets

Download the public external datasets into `dataset/external` with:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ptdr
bash scripts/download_external_datasets.sh
```

This downloads and extracts:

- `EvArEST`
- `Total-Text`
- `CTW1500`
- `TextOCR`
- `IR-LPR`

`ICDAR2019 MLT` is expected under:

- `dataset/external/icdar2019_mlt`

Notes:

- The downloader uses the official CTW1500 evaluation bundle as the fallback source for `gt_ctw1500.zip`, because the old `cloudstor.aarnet.edu.au` test-label host is no longer resolvable.
- `TextOCR` test images are downloaded, but the test JSON is unlabeled and is skipped automatically for training.
- `IR-LPR` plate crops and car images are both supported. The trainer now reads the XML annotations directly for both detection boxes and recognition strings.

Download the official PTDR synthetic recognition corpus with:

```bash
bash scripts/download_ptdr_synth.sh
```

This fetches the `PTDR-SYNTH.zip` archive linked from the upstream PTDR repo README and exposes the extracted flat `gt.txt + images/` dataset at:

- `dataset/synth/ptdr_synth`

## Remote Runners

For simple multi-VM usage on hosts like `visionspot5` through `visionspot9`, use the helper scripts in [`../../scripts/`](../../scripts/):

```bash
scripts/bootstrap_remote.sh visionspot5
scripts/sync_code_to_host.sh visionspot5
scripts/run_dbnetpp_remote.sh visionspot5 dbnetpp-visionspot5-baseline
scripts/run_parseq_remote.sh visionspot5 parseq-visionspot5-baseline
```

Expected SSH setup:

- each host alias resolves through your local `~/.ssh/config`
- each host has `conda` available
- the first bootstrap clones `REPO_URL` into `~/PTDR` (defaults to `remote.origin.url` when available)

The sync script intentionally excludes `dataset/` and `work_dirs/` so repeated experiment iteration only copies code and configs.

## Train DBNet++

Edit [`configs/dbnetpp.yaml`](configs/dbnetpp.yaml) if you want to change domains, checkpoints, or W&B settings, then run:

```bash
python experiments/ptdr/train_dbnetpp.py --config_path experiments/ptdr/configs/dbnetpp.yaml
```

CLI override example:

```bash
python experiments/ptdr/train_dbnetpp.py \
  --config_path experiments/ptdr/configs/dbnetpp.yaml \
  --training.max_epochs 50 \
  --wandb.run_name dbnetpp-scene-50ep
```

Notes:

- `mmocr.base_config` can now be any installed MMOCR text detection config filename that exists under the local MMOCR `textdet` config tree, including DBNet and DBNet++ variants.
- `mmocr.load_from` is left `null` by default. Point it at a DBNet or DBNet++ checkpoint if you want to fine-tune instead of training from scratch.
- A staged DBNet++ curriculum is available via:
  - [`configs/dbnetpp_multidata_clean_stage.yaml`](configs/dbnetpp_multidata_clean_stage.yaml)
  - [`configs/dbnetpp_multidata_hard_finetune_stage.yaml`](configs/dbnetpp_multidata_hard_finetune_stage.yaml)
  - [`../../scripts/run_dbnetpp_two_stage.sh`](../../scripts/run_dbnetpp_two_stage.sh)
- The default curriculum now uses all supported train-only external data, but keeps PTDR dominant and CTW1500 low so the target domain remains the anchor.
- The hard fine-tune stage supports an epoch-based hard-augmentation probability schedule, so you can start clean or very mild and ramp up later while the base PolyLR schedule naturally lowers the learning rate.
- The default config uses only `indoor_text` and `outdoor_text`.
- External datasets are train-only. `val` and `test` remain PTDR-only.
- `train_mix` defines target train fractions. PTDR is the anchor dataset and must remain at or above `train_mix.min_ptdr_fraction`.
- `mlt.scripts` filters ICDAR2019 MLT instances by script label. For English plus Arabic, keep `Arabic` and `Latin`.
- Supported external detection roots:
  - `icdar2019_mlt`
  - `evarest_detection`
  - `totaltext`
  - `ctw1500`
  - `textocr`
  - `ir_lpr_detection`
- Generated manifests land under `experiments/ptdr/manifests/dbnetpp_scene`.
- Each validation epoch logs one random fully-correct validation image if available and one random error case under `work_dirs/dbnetpp_ptdr_scene/val_samples/dbnetpp/`, with predicted polygons rendered on the image.
- Multi-dataset detection config:
  - [`configs/dbnet_multidata_all.yaml`](configs/dbnet_multidata_all.yaml): the single mixed-data DBNet config, now using only `ICDAR2019 MLT` for MLT data.

## Train PARSeq

Edit [`configs/parseq.yaml`](configs/parseq.yaml), then run:

```bash
python experiments/ptdr/train_parseq.py --config_path experiments/ptdr/configs/parseq.yaml
```

CLI override example:

```bash
python experiments/ptdr/train_parseq.py \
  --config_path experiments/ptdr/configs/parseq.yaml \
  --training.batch_size 64 \
  --training.max_epochs 30 \
  --wandb.run_name parseq-scene-bs64
```

Notes:

- The manifest builder writes LMDB datasets under `experiments/ptdr/manifests/parseq_scene`.
- External datasets are train-only. PTDR remains the sole source for validation and test.
- `train_mix` defines target train fractions and enforces a minimum PTDR share via `train_mix.min_ptdr_fraction`.
- MLT roots are consumed through detection annotations and cropped into recognition samples automatically.
- `Total-Text`, `CTW1500`, and `TextOCR` are also converted into recognition crops from their detection annotations.
- `EvArEST` and `IR-LPR` can use dedicated recognition roots through `external_datasets.evarest_recognition` and `external_datasets.ir_lpr_recognition`; if those are not set, the trainer falls back to cropping from their detection roots when possible.
- `ptdr_synth_root` adds the official PTDR synthetic recognition corpus as an optional extra train-only source. The loader supports its native flat `gt.txt + images/` layout directly.
- `include_ptdr_train_in_train_split: false` lets you keep PTDR validation/test while excluding PTDR train from the PARSeq training split, which is useful for synth-only pretraining.
- FATDR is intentionally not used for training because its recognition crops overlap heavily with PTDR by exact crop name and text, which contaminates evaluation.
- With `model.pretrained: true`, the trainer keeps compatible PARSeq pretrained weights and reinitializes the charset-sized `head` and `text_embed` layers for the generated training charset.
- `model.init_from_checkpoint` lets you initialize PARSeq from a previous Lightning checkpoint while still skipping charset-sized tensors that no longer match.
- The recognizer now uses a canonical internal Arabic/Persian target space so equivalent Unicode forms like `ك/ک` and `ي/ی` do not become duplicate classes. See [`RECOGNIZER_CHARSET.md`](RECOGNIZER_CHARSET.md).
- Each validation epoch logs one random correct validation crop if available and one random incorrect crop under `work_dirs/parseq_ptdr_scene/val_samples/parseq/`, with GT and predicted text recorded alongside the image.
- `remove_whitespace` and `normalize_unicode` are disabled by default because PTDR contains Persian strings and multi-token labels that should be preserved.
- `max_label_length: null` means "infer it from the generated train, val, and test splits".
- For remote multi-GPU boxes, `scripts/run_parseq_remote.sh` picks idle GPUs based on `nvidia-smi` and constrains PARSeq to those devices with `CUDA_VISIBLE_DEVICES`. Use `GPU_COUNT=2` or higher if you want to claim more than one idle GPU.
- Multi-dataset recognition config:
  - [`configs/parseq_multidata_all.yaml`](configs/parseq_multidata_all.yaml): the single mixed-data PARSeq config, now using only `ICDAR2019 MLT` for MLT data.

Example multi-dataset override:

```bash
python experiments/ptdr/train_parseq.py \
  --config_path experiments/ptdr/configs/parseq.yaml \
  --external_datasets.icdar2019_mlt dataset/external/icdar2019_mlt \
  --external_datasets.textocr dataset/external/textocr \
  --train_mix.ptdr 0.4 \
  --train_mix.icdar2019_mlt 0.4 \
  --train_mix.textocr 0.2
```

Example PTDR-Synth run:

```bash
python experiments/ptdr/train_parseq.py \
  --config_path experiments/ptdr/configs/parseq_default_ptdr_synth.yaml
```

Two-stage PTDR-Synth then all-data run:

```bash
bash scripts/run_parseq_two_stage.sh
```

This uses:

- [`configs/parseq_ptdr_synth_pretrain.yaml`](configs/parseq_ptdr_synth_pretrain.yaml)
- [`configs/parseq_multidata_all_norm_v2_hard_from_ptdr_synth.yaml`](configs/parseq_multidata_all_norm_v2_hard_from_ptdr_synth.yaml)

## Rotation Handling Experiments

Two standalone training paths are available for handling rotations and hard conditions around the detector/recognizer handoff:

- [`train_crop_rotation_classifier.py`](train_crop_rotation_classifier.py)
  - trains a 4-way per-crop rotation classifier (`0/90/180/270` correction) on the same mixed recognition LMDB used by the hard PARSeq run
  - evaluates on fixed shared PTDR crop val variants: `val`, `val_rot90`, `val_rot180`, `val_rot270`, `val_hard`
  - default config: [`configs/crop_rotation_classifier.yaml`](configs/crop_rotation_classifier.yaml)

- [`train_affine_stn.py`](train_affine_stn.py)
  - trains a pre-detector affine spatial transformer on the mixed hard DBNet train manifest
  - synthetic train pairs are generated with affine-only geometry plus hard photometric corruption
  - synthetic geometry uses reflective borders rather than black fill, so the model is not trained to key off black triangles that will not appear at inference time
  - evaluates on fixed shared PTDR detection val variants: `val`, `val_rot90`, `val_rot180`, `val_rot270`, `val_hard`
  - default config: [`configs/affine_stn.yaml`](configs/affine_stn.yaml)

## End-to-End Evaluation

Run DBNet detection and PARSeq recognition together against the PTDR detection manifests:

```bash
python experiments/ptdr/evaluate_end_to_end.py \
  --dbnet-checkpoint work_dirs/dbnet_r50_ptdr_scene/epoch_1.pth \
  --parseq-checkpoint work_dirs/parseq_ptdr_scene/checkpoints/last.ckpt \
  --split test
```

The evaluator reports:

- end-to-end precision: matched polygon + exact normalized transcript / predicted polygon-text pairs
- end-to-end recall: matched polygon + exact normalized transcript / ground-truth text instances
- keyword recall: per-image unique ground-truth tokens that appear anywhere in the recognized output for that image

Outputs land under `work_dirs/end_to_end_eval/.../` by default and include:

- `summary.json`
- `per_image.jsonl`
- overlay and crop visualization PNGs for the first few evaluated images

For interactive inspection, use [`notebooks/end_to_end_inference.ipynb`](notebooks/end_to_end_inference.ipynb).

### Current Snapshot

Latest end-to-end benchmark:

- detector: `DBNet++` [`work_dirs/dbnetpp_multidata_hard_finetune_stage/best_icdar_hmean_epoch_20.pth`](../../work_dirs/dbnetpp_multidata_hard_finetune_stage/best_icdar_hmean_epoch_20.pth)
- recognizer: hard PARSeq [`work_dirs/parseq_multidata_all_norm_v2_hard/checkpoints/epoch=epoch=017-step=step=0003636.ckpt`](../../work_dirs/parseq_multidata_all_norm_v2_hard/checkpoints/epoch=epoch=017-step=step=0003636.ckpt)
- crop rotation classifier: exhaustive `128x128` MobileNetV2 [`work_dirs/crx4/checkpoints/best-05.ckpt`](../../work_dirs/crx4/checkpoints/best-05.ckpt)
- full report: [`work_dirs/end_to_end_eval/dbnetpp_best_parseqhard_crop_new/combined_summary.json`](../../work_dirs/end_to_end_eval/dbnetpp_best_parseqhard_crop_new/combined_summary.json)

| Split | Variant | Precision | Recall | Keyword recall |
|---|---|---:|---:|---:|
| `val` | baseline | `18.88` | `54.55` | `62.61` |
| `val` | crop | `18.79` | `54.32` | `62.09` |
| `val_rot90` | baseline | `6.91` | `18.61` | `29.93` |
| `val_rot90` | crop | `7.53` | `20.27` | `35.42` |
| `val_rot180` | baseline | `11.76` | `37.93` | `48.80` |
| `val_rot180` | crop | `12.25` | `39.51` | `51.46` |
| `val_rot270` | baseline | `6.17` | `16.31` | `27.19` |
| `val_rot270` | crop | `7.58` | `20.03` | `36.28` |
| `val_hard` | baseline | `29.02` | `39.98` | `50.00` |
| `val_hard` | crop | `29.08` | `40.06` | `50.17` |

Current interpretation:

- `DBNet++ + PARSeq hard` is the strongest detector/recognizer pair we have validated end to end.
- The new crop rotation classifier is useful mainly on `90/180/270` rotated validation splits.
- On clean `val`, crop correction is roughly neutral to slightly negative.
- On `val_hard`, the gain from the crop classifier is real but small.
