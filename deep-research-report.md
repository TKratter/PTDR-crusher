# Scene Text OCR Literature Review With Arabic and Persian Focus

## Executive summary

Scene text OCR (often called *scene text understanding* or *text spotting* when end-to-end) differs from document OCR because text appears with uncontrolled capture conditions: perspective distortions, cluttered backgrounds, varying illumination, motion blur, low resolution, and irregular layouts. Modern systems are typically (a) **two-stage** pipelines—text detection followed by text recognition—or (b) **end-to-end spotters** that jointly localize and transcribe text in one model. citeturn19view0turn45view3

Over roughly the last five years, the field has converged on a few dominant design patterns:

- **Detection** is increasingly segmentation- or contour-representation-driven (e.g., differentiable binarization, progressive kernel expansion, Fourier contour embeddings), with growing adoption of **transformer-based** detectors that regress polygons or control points directly and reduce heuristic post-processing. citeturn38view2turn37view2turn40view0turn41view3  
- **Recognition** has moved from CNN–RNN–CTC baselines to richer **vision–language** models and transformer decoders that improve robustness on irregular text and exploit linguistic priors (explicitly or implicitly). citeturn37view4turn43view0turn44view0turn42view0  
- **Text spotting** (end-to-end detection+recognition) is increasingly **transformer-centric** and explicitly targets *synergy* between detection and recognition, either within unified decoders or via “bridge” designs that combine the benefits of end-to-end and two-step pipelines. citeturn46view2turn46view3turn45view3  

For Arabic and Persian (Farsi), the primary bottleneck remains **data** (scale, diversity, and annotation formats) plus **script-specific challenges**: right-to-left directionality, dot/diacritic sensitivity, fine-grained glyph similarity, ligatures and cursive joining, and mixed-script content (Arabic + Latin digits/English) in real scenes. Recent work has responded by releasing Arabic/Persian-focused datasets and large synthetic corpora, but benchmark coverage is still much thinner than Latin/Chinese. citeturn34view0turn47view1turn35view0turn14view0  

---

## Methods landscape

### Scene text detection

Most strong detectors today output **quadrilaterals or polygons** (rather than axis-aligned boxes). Two high-level families dominate: (1) segmentation-like probability maps followed by structured extraction, and (2) direct regression of contours/vertices (increasingly with transformers). citeturn38view2turn40view0turn41view3turn33view0  

The table below summarizes representative and widely-used detectors (seminal-to-recent), emphasizing models with reproducible numbers and public code/configs.

| Method | Core idea and representation | Typical benchmarks and reported performance (examples) | Primary code / configs |
|---|---|---|---|
| **DBNet** (AAAI 2020) | Segmentation-style detector with **Differentiable Binarization (DB)**: makes thresholding trainable so post-processing is simplified and accuracy improves across shapes. citeturn38view2turn37view0 | ICDAR2015 (IC15) Hmean up to **0.8644** (R50-oCLIP config); Total-Text Hmean **0.8182** (R18). citeturn38view2 | DB configs + downloadable models (MMOCR). citeturn38view2turn39search2 |
| **DBNet++** (TPAMI 2022) | Extends DB with **Adaptive Scale Fusion (ASF)** to improve scale robustness while keeping a simple pipeline. citeturn38view3turn36view0 | ICDAR2015 Hmean up to **0.8882** (R50-oCLIP config). citeturn38view3 | DBNet++ configs + models (MMOCR). citeturn38view3turn36view0 |
| **PSENet** (CVPR 2019) | Produces multiple **kernels** per instance and **progressively expands** them to separate nearby text instances—good for arbitrary shapes and crowded scenes. citeturn37view2turn39search6 | CTW1500 Hmean **0.8037** (R50-oCLIP); ICDAR2015 Hmean **0.8478** (R50-oCLIP). citeturn37view2 | PSENet configs + models (MMOCR). citeturn37view2turn39search2 |
| **FCENet** (CVPR 2021) | Represents text contours in the **Fourier domain** (compact signature) and reconstructs contours via inverse Fourier transform + NMS—targets highly-curved text. citeturn40view0 | CTW1500 Hmean **0.8488**; ICDAR2015 Hmean **0.8528**; Total-Text Hmean **0.8134** (MMOCR configs). citeturn40view0 | FCENet configs + models (MMOCR). citeturn40view0turn39search2 |
| **DPText-DETR** (AAAI 2023) | Transformer detector using **dynamic point queries** (explicit point coordinates) with progressive refinement; reports strong results on polygon benchmarks. citeturn41view3turn36view2 | Reported F-measure: **89.0** (Total-Text), **88.8** (CTW1500), **78.1** (ICDAR19 ArT). citeturn41view3 | Official repo with models and results. citeturn41view3turn36view2 |

**How to interpret “state-of-the-art” claims in detection.** Reported best numbers often depend on (a) annotation protocols and polygon matching, (b) training data cocktail (e.g., SynthText + MLT + LSVT), and (c) test-time resizing and multi-scale inference. The Robust Reading Competition (RRC) ecosystem exists precisely to standardize these evaluations, but leaderboards continue to evolve. citeturn19view0turn33view0turn12search3turn16search21  

### Scene text recognition

Recognition systems are often evaluated on standard cropped-word benchmarks (e.g., IIIT5K, SVT, ICDAR13/15, SVTP, CUTE80). The community increasingly reports **word accuracy** and also **normalized edit distance** (often as *1 − NED*) to reflect near-miss errors. citeturn37view4turn42view0turn47view1  

Representative methods:

| Method | Architecture sketch | Key innovation | Example benchmark numbers (from official repos/configs) |
|---|---|---|---|
| **CRNN** (TPAMI 2016) | CNN feature extractor → BiLSTM sequence model → CTC decoding. citeturn37view4 | Early deep baseline still widely used due to simplicity; handles variable-length without explicit character segmentation. citeturn37view4 | Example results (MMOCR): IIIT5K **80.5**, SVT **81.5**, IC13 **86.5**, IC15 **54.1**, SVTP **59.1**, CUTE80 **55.6**. citeturn37view4 |
| **ABINet** (CVPR 2021) | Vision model + explicit language model trained end-to-end; iterative correction. citeturn36view3turn43view0 | “Read like humans” via bidirectional cloze-style language representation (BCN) and iterative refinement. citeturn36view3turn43view0 | Repo-reported averages over 6 benchmarks: **91.4** (ABINet-SV) and **92.7** (ABINet-LV), with per-dataset breakdown (IC13/SVT/IIIT/IC15/SVTP/CUTE). citeturn43view0 |
| **VisionLAN** (ICCV 2021) | Single recognizer that fuses visual + linguistic cues without an extra LM. citeturn44view0 | “Two to one” unified visual-language modeling; includes evaluations on standard 6-benchmark suite. citeturn44view0 | Paper vs. implementation on 6 benchmarks (example): IIIT5K **95.9**, IC13 **96.3**, SVT **90.7**, IC15 **84.1**, SVTP **85.3**, CUTE **88.9**. citeturn44view0 |
| **PARSeq** (ECCV 2022) | Transformer-based sequence modeling; supports AR and iterative refinement variants. citeturn42view0turn36view4 | Permuted autoregressive training / strong decoding behavior; repo emphasizes reproducible benchmarking and reports both Accuracy and 1−NED. citeturn42view0turn36view4 | Example (lowercase alnum): IIIT5K **99.00**, SVT **97.84**, IC13 **98.13**, IC15 **89.22**; Combined accuracy **95.95** and 1−NED **98.78**. citeturn42view0 |
| **ViTSTR** (ICDAR 2021) | Vision Transformer for efficient STR. citeturn10search0turn10search4 | Emphasizes speed/efficiency tradeoffs; reports competitive accuracy with fewer params/FLOPs and faster inference. citeturn10search0turn10search4 | Reports comparisons and speedup claims on standard STR benchmarks (see paper + repo). citeturn10search0turn10search4 |
| **SVTR** (IJCAI 2022) | “Single visual model” (patch-wise tokenization) removing explicit sequential modeling. citeturn11search0turn37view3 | Dispenses with RNN/transformer text decoders by designing global/local mixing blocks over visual tokens. citeturn11search0turn37view3 |
| **TrOCR** (2021) | Transformer-based OCR with pretrained image + text transformers. citeturn10search3turn10search7 | Shows strong gains from large-scale pretraining + fine-tuning; frequently used as a transferable recognizer backbone beyond document OCR. citeturn10search3turn10search7 |

### End-to-end scene text spotting

Text spotting evaluates both localization and transcription jointly (often with lexicon-free and lexicon-based settings). Recent systems explicitly address **detection–recognition synergy**, reducing ad-hoc post-processing and improving end-to-end consistency. citeturn45view3turn46view2turn46view3turn33view0  

| Method | High-level design | Reported performance examples (detection & E2E) | Code |
|---|---|---|---|
| **ESTextSpotter** (ICCV 2023) | Transformer spotter with **task-aware queries** and a vision-language communication module to tighten synergy. citeturn9search2turn46view2turn45view1 | Model zoo examples: Total-Text Det-F1 **90.0**, E2E-None **80.9**, E2E-Full **87.1**; CTW1500 Det-F1 **89.9**, E2E-None **65.0**, E2E-Full **83.9**; ICDAR2015 Det-F1 **91.4**, E2E-S **88.5**, E2E-W **83.1**, E2E-G **78.1**. citeturn46view2 | Official repo with weights. citeturn45view1turn46view2 |
| **Bridge Text Spotting** (CVPR 2024) | “Bridge” pipeline combining end-to-end and two-step advantages (explicitly stated in the repo). citeturn46view3turn45view2 | Reported: Total-Text Det-F1 **89.2**, E2E-None **83.3**, E2E-Full **88.3**; CTW1500 Det-F1 **89.0**, E2E-None **69.8**, E2E-Full **83.9**; ICDAR2015 Det-F1 **90.5**, E2E-S **89.1**, E2E-W **84.2**, E2E-G **80.4**. citeturn46view3 | Official repo with weights. citeturn46view3turn45view2 |
| **SwinTextSpotter v2** (2024) | End-to-end transformer spotter improving synergy with recognition-conversion + recognition-alignment modules; emphasizes arbitrary-shaped text without extra rectification or character-level labels. citeturn45view3turn9search10 | Paper claims state-of-the-art on multilingual benchmarks and points to code release. citeturn45view3turn9search10 | Linked from the paper. citeturn45view3turn9search10 |
| **ABCNet v2** (TPAMI 2022) | Bezier-curve representation for arbitrary-shaped text spotting with structured outputs. citeturn9search3turn9search7 | Paper positioning and representation contributions; see paper for benchmark tables (e.g., Total-Text/CTW1500/ICDAR). citeturn9search3turn9search7 |
| **ArT RRC baseline / results context** | The ArT challenge report provides top scores and standardized tasks for detection, recognition, spotting on arbitrary-shaped text. citeturn32view0turn33view0 | Top-performing scores reported in the challenge report abstract (task-specific). citeturn32view0 | Challenge kit + dataset link. citeturn32view0turn33view0 |

---

## Benchmark datasets and annotations

Scene text research relies heavily on a small set of benchmarks that define “standard” evaluation. Important differences include: whether text is *focused* vs *incidental*, whether annotations are word-level or line-level, whether geometry is quadrilateral vs polygon, and whether the dataset targets multilingual scripts. citeturn19view0turn33view0turn28view2turn47view1  

### Dataset comparison table

The table focuses on commonly cited scene-text datasets (including those you listed), emphasizing **language/script coverage**, **annotation format**, and **typical metrics**.

| Dataset | Primary task(s) | Size (headline) | Language / script coverage | Annotation format | Typical evaluation metrics |
|---|---|---:|---|---|---|
| **ICDAR2015 Incidental Scene Text (IC15)** | detection, recognition, end-to-end | 1,670 images total; 1,000 train / 500 test (+ 170 private); 17,548 annotated regions. citeturn19view0 | Mainly Latin-script “care” regions; non-Latin often marked “do not care” in this edition. citeturn19view0 | Word-level **quadrilaterals** + Unicode transcription; “care / do-not-care”. citeturn19view0 | Detection commonly IoU-based matching (e.g., IoU ≥ 0.5) with precision/recall/Hmean. citeturn19view0 |
| **ICDAR2017 MLT (RRC-MLT 2017)** | detection + script identification (and supports building recognizers) | 18,000 images. citeturn18view0 | 9 languages: Arabic, Bangla, Chinese, English, French, German, Italian, Japanese, Korean. citeturn18view0 | Widely used with **quadrilateral** ground truth for multi-oriented text. citeturn33view0 | Hmean for detection; script-ID metrics depend on task definition (RRC). citeturn12search3turn18view0 |
| **ICDAR2019 MLT (RRC-MLT 2019)** | detection, script classification, joint, end-to-end detection+recognition | 20,000 real images; 277,000 synthetic images. citeturn14view0turn16search7 | 10 languages incl. **Arabic** and Devanagari/Hindi; 2,000 images per language target (with multilingual co-occurrence). citeturn14view0 | Detection + end-to-end tasks (RRC protocols). citeturn14view0 | Task-specific RRC metrics: detection Hmean; end-to-end combines localization + transcription. citeturn14view0turn16search7 |
| **Total-Text** | detection (curved + mixed), (optionally spotting) | 1,555 images. citeturn29view0 | English; designed to include horizontal, multi-oriented, and curved text. citeturn29view0turn31view1 | Word-level; provides ground truth in `.txt` and pixel-level ground truth (dataset repo notes updates). citeturn31view2turn31view3 | Precision/Recall/F-score (often DetEval variants) or polygon IoU depending on protocol. citeturn31view0turn33view0 |
| **CTW1500 (SCUT-CTW1500)** | detection (curved, long text) | 1,500 images; >10k annotations; 1,000 train / 500 test. citeturn29view3 | Mixed English/Chinese noted in common usage guidance. citeturn31view1 | Polygon-based curved text annotations; introduced as polygon-based curve text dataset. citeturn29view3turn33view0 | Polygon IoU → precision/recall/Hmean; curved-text-specific protocols. citeturn29view3turn33view0 |
| **ArT (RRC-ArT, ICDAR2019)** | detection, recognition, spotting (arbitrary-shaped) | 10,166 images total; 5,603 train / 4,563 test. citeturn33view0 | Chinese + Latin scripts (explicitly noted). citeturn33view0 | Tight **polygon** outputs; polygons may have 4/8/10/12 vertices; Chinese line-level, Latin word-level; transcriptions + language type; “do not care” for illegible/symbols. citeturn33view0turn32view0 | IoU-based evaluation (multiple thresholds reported; Hmean used for ranking). citeturn33view0turn32view0 |
| **COCO-Text** | detection, recognition, script/type classification | >63k images; >173k text annotations. citeturn28view2 | Multi-script (script labels provided per instance). citeturn28view2 | Bounding boxes + typed attributes: printed/handwritten, legible/illegible, script; transcriptions for legible text. citeturn28view2 | Task-dependent; often detection precision/recall/Hmean and recognition accuracy/edit distance for legible subsets. citeturn28view2 |
| **SynthText (SynthText in the Wild)** | synthetic **pretraining** for detection/recognition | 800,000 synthetic images; ~10 word instances/image; word- and character-level boxes. citeturn28view3 | Synthetic rendering supports broad Latin vocabulary by default; widely adapted to other scripts via custom generators. citeturn28view3turn34view2 | Automatically generated word + character bounding boxes (and rendered text). citeturn28view3 | Not a benchmark in the same sense; used for pretraining, then fine-tuned on real benchmarks. citeturn28view3turn13search0 |
| **ReCTS-25k (ICDAR2019-ReCTS)** | Chinese signboard: character/line recognition, line detection, end-to-end | 25,000 signboard images; lines + characters annotated. citeturn25search0turn20view0 | Chinese-focused; addresses large character set and diverse layouts. citeturn20view0 | Text line + character locations + transcriptions; multi-GT evaluation proposed for ambiguity. citeturn20view0 | Task-specific: detection metrics + recognition accuracy; multi-GT handling for fair evaluation. citeturn20view0 |
| **TextOCR** | large-scale recognition & detection in complex scenes | (Large COCO-based text dataset; see paper for exact counts and splits.) citeturn4search2turn4search12 | Multi-language varies by COCO image content; designed for diverse “text-in-the-wild”. citeturn4search2turn25search23 | Dense word-level annotations on COCO images (paper describes). citeturn4search2 | Detection Hmean; recognition accuracy / edit-distance variants; increasingly used to study data scaling. citeturn25search23turn4search2 |

---

## Arabic and Persian dataset coverage

Arabic and Persian are both right-to-left scripts, but Persian introduces additional orthographic and typographic characteristics (fonts, letter variants, common mixed numerals) that make “Arabic-trained” models non-trivially transferable. citeturn47view1turn35view3turn34view0  

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["EvArEST Arabic English scene text dataset sample images","PTDR Persian scene text dataset sample images","FATR Persian text recognition dataset sample images in the wild","ICDAR 2019 MLT Arabic scene text sample images"],"num_per_query":1}

### Datasets that include Arabic scene text

**ICDAR2017 MLT (RRC-MLT 2017)** is one of the most practically useful “Arabic-included” benchmarks because it provides *Arabic in real scene images* alongside eight other languages, enabling detector training that must generalize across scripts. citeturn18view0turn33view0  
- Size and languages: 18,000 images covering (among others) **Arabic**. citeturn18view0  
- Annotation geometry: commonly used with quadrilateral annotations for detection in this benchmark family. citeturn33view0  
- Practical value: good for multilingual detection and script-ID; weaker if you need dense Arabic-only coverage for recognition. citeturn18view0turn46view2  

**ICDAR2019 MLT (RRC-MLT 2019)** extends the multilingual direction with a much larger standardized package and adds a substantial synthetic counterpart. citeturn16search7turn14view0  
- Size and languages: 20,000 real images across 10 languages including **Arabic**, plus 277,000 synthetic images aligned to the same language set. citeturn14view0turn16search7  
- Tasks: detection, cropped-word script classification, joint detection+script classification, and end-to-end detection+recognition. citeturn16search7turn14view0  
- Strength: supports multilingual end-to-end evaluation without requiring Arabic-only datasets; limitation is that Arabic is only one portion of a multilingual mix. citeturn14view0turn16search7  

**EvArEST (Everyday Arabic–English Scene Text)** is a targeted Arabic resource with both detection and recognition subsets plus Arabic/English bilinguality, which matters for signage in many regions. citeturn34view0turn14view3  
- Detection subset: 510 images; each **word** annotated with a four-point polygon (clockwise) and a language tag. citeturn34view0  
- Recognition subset: 7,232 cropped-word images (Arabic + English) with filename→transcription ground truth. citeturn34view0  
- Synthetic data: ~200k synthetic images with segmentation maps are linked from the dataset repository. citeturn34view0  
- Benchmarking: the associated paper positions EvArEST as a dataset for Arabic and bilingual recognition analysis and provides results for follow-on work. citeturn14view3turn34view0  

### Datasets that include Persian scene text

**PTDR (Persian Scene + Document Text Detection and Recognition, 2025)** is currently the most clearly specified, benchmark-style dataset in the sources reviewed here, with explicit scene categories, train/test counts, and multiple export formats. citeturn47view1turn35view3  
- Total size: 2,197 images for detection (1,651 train / 546 test). citeturn47view1  
- Recognition size: 18,424 train word patches / 5,882 test word patches. citeturn47view1turn47view3  
- Annotations: quadrilateral bounding boxes (4 vertices) and transcription; annotated using VIA; exports include ICDAR-style `.txt`, COCO-style `.json`, and YOLO formats (including rotated rectangles). citeturn47view1turn47view3  
- Synthetic pretraining: PTDR-Synth is ~200k word images, generated across >200 Persian fonts. citeturn47view0turn47view1  
- Benchmark result signals: the paper reports that detection models pretrained on multilingual datasets (e.g., ICDAR17-MLT) can still perform reasonably on PTDR detection; it also provides WRA / normalized metrics definitions and comparative evaluations. citeturn35view3turn47view0turn47view1  

**FATR (Persian Text Recognition in Wild Images, 2025)** focuses on Persian recognition and explicitly reports that SOTA models degrade on Persian without appropriate training data. citeturn35view0turn34view1  
- Synthetic companion: FATR-Synth “over 200,000” cropped word images for Persian pretraining. citeturn35view0  
- Public access: the journal page points to the accompanying repository (FATDR) for code/data. citeturn35view0turn34view1  
- Gap: the public abstract-style page does not state the exact number of real cropped-word images; consult the paper PDF or data package for precise counts. citeturn35view0turn34view1  

**Persian synthetic scene-text generators / datasets** also exist and are used to emulate SynthText-style pretraining for Persian-specific typography. citeturn34view2turn47view1  
- Example (synthetic): a publicly described Persian synthetic dataset provides (a) cropped word images for recognition and (b) full scene images for detection and end-to-end, along with sample images and download links. citeturn34view2  

### Multilingual datasets that cover Arabic but not Persian, and remaining gaps

- MLT benchmarks (2017/2019) explicitly include **Arabic**, but **Persian** is not among the listed language sets—making Persian OCR more reliant on dedicated datasets like PTDR/FATR or new multilingual expansions. citeturn18view0turn14view0turn47view1  
- COCO-Text provides a *script label* per instance and is broad “text-in-the-wild,” but it is not a targeted Arabic/Persian benchmark and does not guarantee adequate coverage for either language. citeturn28view2turn47view1  
- Arabic resources exist but are often (a) smaller, (b) domain-specific (e.g., signage subsets), or (c) optimized for recognition only rather than full detection→recognition evaluation. citeturn34view0turn14view4turn14view3  
- Persian benchmark resources are improving (PTDR, FATR), but the overall ecosystem still lacks a single widely adopted *pure-scene* Persian “ICDAR-grade” benchmark with large outdoor diversity comparable to Latin/Chinese datasets. citeturn47view1turn35view0  

---

## Evaluation, training recipes, and pipelines

### Evaluation metrics

**Detection**: Standard evaluations match predicted regions to ground truth using an overlap criterion (often IoU ≥ 0.5), then compute Precision, Recall, and Hmean (harmonic mean). ICDAR2015’s report explicitly describes an IoU threshold-based evaluation for localization tasks. citeturn19view0turn33view0  

**Spotting (end-to-end)**: End-to-end benchmarks combine localization correctness with transcription correctness; leaderboards frequently report multiple settings (lexicon-free vs lexicon-based, or multiple lexicon granularities). citeturn19view0turn46view2turn46view3turn33view0  

**Recognition**: Common metrics include:
- **Word Recognition Accuracy (WRA)**, used explicitly in Persian benchmarks like PTDR and in the FATR paper description. citeturn47view0turn35view0  
- **Normalized edit distance** (often reported as *1 − NED*) to reflect near-miss transcription errors; PARSeq’s repo demonstrates reporting both accuracy and 1−NED over standard benchmarks. citeturn42view0turn47view0  

### Pretraining with synthetic data

Synthetic data is a central enabler for scene text OCR because large-scale manually labeled text-in-the-wild is expensive. Two long-standing pillars are:

- **SynthText in the Wild** (800k images) providing word- and character-level boxes, created to train detection/localization networks at scale. citeturn28view3  
- **MJSynth / Synth90k** (9 million word images covering 90k English words), published as synthetic recognition data and released via the Oxford text recognition data page. citeturn13search2turn13search7  

These datasets are also reflected in standardized STR benchmarking practice (e.g., unified evaluation and training set discussions). citeturn13search0turn13search4turn37view4  

For Arabic/Persian, multiple works explicitly introduce **script-specific synthetic generators** or large synthetic word corpora (EvArEST synthetic set; PTDR-Synth; FATR-Synth), motivated by the scarcity of labeled real data and by typography complexity. citeturn34view0turn47view0turn35view0  

### Typical pipelines

```mermaid
flowchart TD
  A[Input image: natural scene] --> B[Preprocess: resize / normalize / denoise]
  B --> C{Approach family}

  C -->|Two-stage pipeline| D[Text detection]
  D --> E[Post-process: NMS / polygon grouping / thresholding]
  E --> F[Crop / rectify (optional): perspective, TPS, RoIRotate]
  F --> G[Text recognition: CNN-RNN-CTC or Transformer/Vision-Language]
  G --> H[Language-aware decoding (optional): LM, lexicon, beam search]
  H --> I[Output: boxes/polygons + transcriptions]

  C -->|Segmentation-based detection| D2[Predict text maps: region/affinity/kernels/DB]
  D2 --> E2[Convert maps to instances: connected components / expansion / binarization]
  E2 --> F
  C -->|End-to-end text spotting| J[Unified detector+recognizer (Transformer/spotter)]
  J --> I
```

This flow consolidates what is explicitly described across (a) ICDAR-style evaluation definitions (localize then recognize), (b) segmentation-based detectors like DB/PSENet/FCENet, and (c) end-to-end spotters that jointly predict localization and sequences. citeturn19view0turn38view2turn37view2turn40view0turn46view2turn46view3  

---

## Open challenges and promising directions for Arabic and Persian scene text

### Data scale, diversity, and annotation formats remain the primary bottleneck

Even papers introducing Arabic/Persian datasets emphasize the overall scarcity of standardized, comprehensive, openly accessible benchmarks—particularly for detection + recognition together in unconstrained scenes. citeturn14view4turn47view1turn35view0turn34view0  
A practical direction is to grow **ICDAR/RRC-style** evaluation suites for Persian *scene-only* images (not mixed with documents) with polygon annotations for curved/long text and strong multilingual co-occurrence statistics (Arabic–Persian–English–digits). citeturn33view0turn14view0turn47view1  

### Script-specific error modes: dots, ligatures, and bidirectionality

PTDR highlights Persian-specific complexities such as dot-based disambiguation (“Noghteh”), substantial glyph overlap, and the need for polygon precision for some letter shapes. citeturn35view3turn47view0turn47view1  
This motivates research into:
- **Higher-resolution or super-resolution-assisted recognition** for dot/diacritic fidelity (especially in low-res signage). citeturn47view1turn25search23  
- **Representation learning that is sensitive to small dot patterns** (e.g., multi-scale tokenization or explicit diacritic heads), potentially leveraging the strong vision–language paradigms used in recent STR. citeturn47view0turn43view0turn42view0  
- **Bidirectional layout handling** where Arabic/Persian words co-occur with Latin words and numbers (explicitly a concern in Arabic–English datasets like EvArEST). citeturn34view0turn14view3  

### Better synthetic-to-real transfer for right-to-left scripts

Arabic/Persian work frequently pairs real datasets with large synthetic corpora (EvArEST synthetic set; FATR-Synth; PTDR-Synth) to address data scarcity, but synthetic-to-real gaps persist. citeturn34view0turn35view0turn47view0  
Promising directions include:
- **Font- and rendering-diverse synthesis** (PTDR-Synth explicitly mentions >200 fonts) and style/illumination augmentations that better approximate real storefront and street-view conditions. citeturn47view0turn47view1  
- **Domain adaptation** or self-training using unlabeled real images in target geographies and sign styles, with careful noise filtering in pseudo-labeling. citeturn47view1turn13search11  

### End-to-end synergy and multilingual modeling

Recent end-to-end spotters emphasize explicit synergy mechanisms between detection and recognition (ESTextSpotter; SwinTextSpotter v2) and hybrid “bridge” training that retains two-step strengths while enabling end-to-end optimization. citeturn46view2turn46view3turn45view3  
For Arabic/Persian, this suggests:
- **Multilingual spotters with script-aware decoding** and strong shared representations, trained on multilingual datasets that include Arabic (MLT) and extended via Persian datasets (PTDR) plus synthetic corpora. citeturn14view0turn47view1turn46view2  
- **Evaluation protocols that reflect real deployments**, e.g., mixed-script signage, brand names, and numerals, rather than pure-language subsets. citeturn34view0turn47view1turn14view0  

---

## Recommended primary sources to prioritize

The following are high-value “start here” sources for an Arabic/Persian-focused scene text OCR literature review, prioritizing official papers, benchmark reports, and code repositories.

**Benchmark ecosystems and evaluation**
- Robust Reading Competition platform & reports (ICDAR/RRC standardization context). citeturn12search3turn16search21turn19view0  
- ArT challenge report (dataset composition, polygon formats, evaluation thresholds, and official results). citeturn32view0turn33view0  

**Detection methods (with reproducible configs)**
- DBNet + DBNet++ (MMOCR configs and numbers; original implementation is also widely referenced). citeturn38view2turn38view3turn36view0  
- PSENet and FCENet (MMOCR configs, benchmark numbers). citeturn37view2turn40view0  
- DPText-DETR (official repo; strong transformer detector results on polygon benchmarks). citeturn41view3turn36view2  

**Recognition methods**
- CRNN baseline + standardized evaluation dataset counts (MMOCR STR docs). citeturn37view4  
- ABINet, VisionLAN, PARSeq (official repos provide reproducible benchmark tables and pretrained models). citeturn43view0turn44view0turn42view0  
- Synthetic recognition data sources: MJSynth/Synth90k official page and classic synthetic-data paper. citeturn13search7turn13search2  

**End-to-end spotting**
- ESTextSpotter and Bridge Text Spotting (official repos include model zoos and E2E metrics). citeturn46view2turn46view3  
- SwinTextSpotter v2 paper (synergy mechanisms; multilingual scope; code pointer). citeturn45view3turn9search10  

**Arabic and Persian datasets**
- EvArEST dataset repository + accompanying analysis paper (Arabic/English detection + recognition + synthetic). citeturn34view0turn14view3  
- PTDR (dataset statistics, formats, metrics, synthetic PTDR-Synth; explicit benchmark framing). citeturn47view1turn35view3turn47view0  
- FATR / FATDR (Persian recognition benchmark framing + synthetic dataset claim and access pointer). citeturn35view0turn34view1  
- Persian synthetic scene-text dataset repo (useful for controlled pretraining and tooling). citeturn34view2