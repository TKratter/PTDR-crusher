# Recognizer Charset and Canonicalization

This document explains the internal text space used by the PARSeq recognizer in this repo:

- what labels are canonicalized before training
- which characters are treated as the same class
- which characters remain separate classes
- what the recognizer's effective output classes are

This is the target space used by:

- PARSeq training-manifest building in [`build_recognition_manifest.py`](build_recognition_manifest.py)
- shared recognition eval-set generation in [`build_recognition_eval_variants.py`](build_recognition_eval_variants.py)
- end-to-end OCR evaluation in [`end_to_end_utils.py`](end_to_end_utils.py)

## Goal

The recognizer should work for:

- Persian
- Arabic
- English

while avoiding fake duplicate classes caused only by Unicode form differences.

Example of a bad duplicate:

- `ك` and `ک`
- `ي` and `ی`

Those are separate Unicode code points, but for this OCR training setup we do not want the model to waste capacity learning them as separate target classes.

## Canonicalization Rules

Canonicalization happens before:

- building the train/val/test recognition manifests
- building the shared recognition eval LMDBs
- comparing text in end-to-end evaluation and keyword recall

### 1. Equivalent Arabic/Persian letters are merged

These are normalized to a single internal target form:

- `ك -> ک`
- `ي -> ی`

The chosen internal target is the Persian form.

### 2. Digit variants are merged

All digit variants are mapped to one internal digit set.

Current target:

- Persian digits `۰۱۲۳۴۵۶۷۸۹`

So both of these normalize into the same internal representation:

- Arabic-Indic digits `٠١٢٣٤٥٦٧٨٩`
- Persian digits `۰۱۲۳۴۵۶۷۸۹`

ASCII digits `0123456789` are part of the recognizer charset as their own classes and remain available for Latin/English data.

### 3. Unicode normalization is not used as a broad rewrite step

We do not apply aggressive text rewriting that would collapse genuinely different letters.

This is deliberate. The recognizer should not silently erase meaningful Arabic distinctions just to reduce charset size.

## Characters That Stay Separate

These are not collapsed by the canonicalization layer:

- `ى` vs `ی`
- `ة` vs `ه`
- `ا` vs `أ` vs `إ` vs `آ`
- `ؤ` vs `و`
- `ئ` vs `ی`

These may be related linguistically in some contexts, but they are not treated as the same OCR class here.

## Effective Internal Recognizer Classes

The base internal recognizer charset is stored in:

- [`charsets/parseq_allowed_base_charset.txt`](charsets/parseq_allowed_base_charset.txt)

That charset is:

```text
 !#%&'()*+-./0123456789:=@ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz«»،؟ءآأئابتثجحخدذرزسشصضطظعغفقلمنهوَ٪پچژکگی۰۱۲۳۴۵۶۷۸۹‍
```

This is the canonical target space the recognizer is anchored to.

## Class Groups

### English / Latin

Uppercase:

```text
ABCDEFGHIJKLMNOPQRSTUVWXYZ
```

Lowercase:

```text
abcdefghijklmnopqrstuvwxyz
```

ASCII digits:

```text
0123456789
```

### Persian / Arabic script classes kept in the base charset

```text
ءآأئابتثجحخدذرزسشصضطظعغفقلمنهوپچژکگی
```

Also present as separate classes:

```text
َ
```

and:

```text
‍
```

where:

- `َ` is Arabic fatha
- `‍` is zero-width joiner

### Persian digits

```text
۰۱۲۳۴۵۶۷۸۹
```

### Punctuation / symbols kept in the base charset

```text
 !#%&'()*+-./:=@_«»،؟٪
```

## What Is No Longer a Separate Class

Under the current canonicalization policy, these are not separate recognizer classes anymore:

- `ك`
- `ي`
- Arabic-Indic digits `٠١٢٣٤٥٦٧٨٩`

They are normalized before the model sees them as labels.

## Why This Helps

Without canonicalization, mixed Arabic/Persian external datasets can introduce duplicate label targets such as:

- `ك` and `ک`
- `ي` and `ی`

That hurts the recognizer in two ways:

- it expands the output space unnecessarily
- it teaches the model to distinguish Unicode variants that we do not actually want as separate OCR classes

Canonicalization keeps the task smaller and more coherent while still allowing Arabic and Persian data to contribute to training.

## Important Boundary

This document describes the internal recognizer target space only.

It does not define any surface-form post-processing policy for output display.

That means:

- training labels are canonicalized
- evaluation matching is canonicalized
- end-to-end keyword recall is canonicalized
- recognizer outputs are not post-converted back into Arabic or Persian display variants

If we later want display-time formatting rules, that should be added as a separate post-processing layer, not mixed into the model target space.

## Source of Truth in Code

Canonicalization logic lives in:

- [`text_normalization.py`](text_normalization.py)

Training-time label normalization is applied in:

- `normalize_label_for_charset_policy` in [`build_recognition_manifest.py`](build_recognition_manifest.py)

Evaluation-time text normalization is applied in:

- `normalize_eval_text` in [`end_to_end_utils.py`](end_to_end_utils.py)

Charset-policy defaults are defined in:

- [`config_schema.py`](config_schema.py)
