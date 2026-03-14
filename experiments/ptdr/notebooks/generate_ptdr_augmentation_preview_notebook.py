from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code_cell(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        markdown_cell(
            """
            # PTDR Augmentation Preview

            This notebook helps us inspect **PTDR-style detection images** under the kinds of distortions we expect at deployment time:

            - strong rotation, including upside down text
            - perspective / camera-angle changes
            - low light, glare, and shadow
            - motion blur and compression

            Each view draws:

            - **green polygons**: text polygons
            - **orange dashed boxes**: axis-aligned boxes derived from the polygons

            The goal here is not to pick perfect augmentation parameters yet. It is to quickly see which distortions still look realistic for your target photos before we port any of them into training.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            from IPython.display import display

            REPO_ROOT = Path.cwd().resolve()
            repo_root_hint = os.environ.get("PTDR_REPO_ROOT")
            candidate_roots = [REPO_ROOT, *REPO_ROOT.parents]
            if repo_root_hint:
                candidate_roots.append(Path(repo_root_hint).expanduser())
            for candidate in candidate_roots:
                if (candidate / "experiments/ptdr/augmentation_preview.py").exists():
                    REPO_ROOT = candidate
                    break
            else:
                raise RuntimeError("Could not locate the PTDR repo root from the current notebook working directory.")

            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))

            from experiments.ptdr.augmentation_preview import (
                DEFAULT_PRESETS,
                choose_samples,
                load_ptdr_detection_samples,
                preview_preset_across_samples,
                preview_sample_variants,
            )

            plt.rcParams["figure.facecolor"] = "white"
            plt.rcParams["axes.facecolor"] = "white"
            print(f"Repo root: {REPO_ROOT}")
            """
        ),
        code_cell(
            """
            # Main controls
            SPLIT = "train"
            INCLUDE_DOMAINS = None
            SAMPLE_COUNT = 4
            RANDOM_SEED = 42
            SHOW_BBOX = True
            SHOW_TEXT = False
            MAX_LABELS = 25

            # Pick a sample from the selected subset for the detailed side-by-side view.
            SAMPLE_INDEX = 0

            # Start from the built-in suggestions and edit them freely.
            PRESETS = [
                dict(DEFAULT_PRESETS[0]),
                dict(DEFAULT_PRESETS[1]),
                dict(DEFAULT_PRESETS[2]),
                dict(DEFAULT_PRESETS[3]),
                dict(DEFAULT_PRESETS[4]),
            ]

            # Example of how to tweak a preset:
            # PRESETS[2]["perspective_strength"] = 0.24
            # PRESETS[3]["motion_blur_kernel"] = 17

            # Pick one preset here to compare across multiple images.
            CROSS_SAMPLE_PRESET_INDEX = 2
            """
        ),
        code_cell(
            """
            samples = load_ptdr_detection_samples(
                REPO_ROOT,
                split=SPLIT,
                include_domains=INCLUDE_DOMAINS,
            )
            selected_samples = choose_samples(samples, count=SAMPLE_COUNT, seed=RANDOM_SEED)

            print(f"Loaded {len(samples):,} PTDR {SPLIT} detection images.")
            print(f"Selected {len(selected_samples)} samples with seed={RANDOM_SEED}.")
            for idx, sample in enumerate(selected_samples):
                print(f"[{idx}] {sample.domain} :: {sample.image_path.name} :: {len(sample.instances)} instances")

            print("\\nActive presets:")
            for idx, preset in enumerate(PRESETS):
                print(f"[{idx}] {preset['name']}: {preset}")
            """
        ),
        code_cell(
            """
            sample = selected_samples[SAMPLE_INDEX]
            fig = preview_sample_variants(
                sample,
                presets=PRESETS,
                seed=RANDOM_SEED,
                columns=3,
                show_bbox=SHOW_BBOX,
                show_text=SHOW_TEXT,
                max_labels=MAX_LABELS,
            )
            display(fig)
            plt.close(fig)
            """
        ),
        code_cell(
            """
            for idx, sample in enumerate(selected_samples):
                print(f"\\nSample {idx}: {sample.domain} / {sample.image_path.name}")
                fig = preview_sample_variants(
                    sample,
                    presets=PRESETS,
                    seed=RANDOM_SEED + idx * 101,
                    columns=3,
                    show_bbox=SHOW_BBOX,
                    show_text=SHOW_TEXT,
                    max_labels=MAX_LABELS,
                )
                display(fig)
                plt.close(fig)
            """
        ),
        code_cell(
            """
            comparison_preset = PRESETS[CROSS_SAMPLE_PRESET_INDEX]
            fig = preview_preset_across_samples(
                selected_samples,
                preset=comparison_preset,
                seed=RANDOM_SEED,
                show_bbox=SHOW_BBOX,
                show_text=SHOW_TEXT,
                max_labels=MAX_LABELS,
            )
            display(fig)
            plt.close(fig)
            """
        ),
        markdown_cell(
            """
            ## What To Look For

            Good augmentation candidates usually keep the scene **plausible** while still making reading harder.

            Signs a preset is probably useful:

            - text is still legible to a human, but noticeably harder
            - polygons still align with visible text regions
            - the distortion matches the way a real camera would fail

            Signs a preset is probably too aggressive:

            - text becomes unreadable even for a human
            - polygons end up mostly outside the visible text after the warp
            - lighting effects look synthetic rather than camera-like

            Once you know which presets look realistic, we can wire the relevant ones into DBNet and PARSeq training separately.
            """
        ),
    ]
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }
    return notebook


def main() -> int:
    output_path = Path(__file__).resolve().parent / "ptdr_augmentation_preview.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    with output_path.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
