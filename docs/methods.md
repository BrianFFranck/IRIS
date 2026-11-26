# Methods (draft)

Working draft of the Methods section, written from the code and `experiments.md`. Every number
here must match a logged run; `[ref]` marks a citation to add, `[TBD]` a result still to come.

## 1. Data

**Development data.** We used the training set of the PANDA challenge (Prostate cANcer graDe
Assessment) [ref], 10,616 H&E-stained prostate core-needle biopsy whole-slide images (WSIs) from
two centres: Radboud University Medical Center (Nijmegen, the Netherlands) and Karolinska
Institutet (Stockholm, Sweden). Each slide has one ISUP grade group (0 = benign, 1 to 5). One
slide whose Gleason score (4+3) contradicts its ISUP grade (2) was removed.

**Locked test set.** Before any experiment, 15% of the slides (1,593) were set aside as a test
set, stratified by centre and ISUP grade (seed 42). These slides were never read during
development; all results below are on the remaining 9,022 development slides (Radboud 4,386,
Karolinska 4,636). One development slide yielded no tissue tiles and has no features, so
analyses use 9,021 slides.

**Cross-validation.** The development slides were split into 5 folds, stratified by centre and
ISUP grade (seed 42). Every model is trained 5 times, each time validated on the fold it did not
train on, so every development slide receives exactly one out-of-fold prediction.

**External data.** [Section 9.]

## 2. Tissue detection and tiling

Tissue was detected on the lowest-resolution level of each slide (about 8 µm per pixel) using
the HSV saturation channel: pure-white padding (mean RGB ≥ 250) and pen ink (green or blue hue,
or dark grey) were excluded, an Otsu threshold was computed on the remaining pixels and capped at
0.08 so that pale tissue is kept, and the mask was cleaned with a morphological closing (disk of
radius 2), opening (radius 1) and removal of objects under 64 pixels.

Slides were cut into a non-overlapping grid of 256 × 256-pixel tiles at full resolution
(level 0: 0.486 µm/px for Radboud, 0.503 µm/px for Karolinska, about 20× magnification; each
tile covers about 128 × 128 µm). A tile was kept if at least 25% of it was tissue. Only tile
coordinates were stored; pixels were read on the fly. An earlier configuration at about 5×
(level 1) was used for initial experiments.

## 3. Tile features

Each tile was encoded by a frozen encoder; no encoder was fine-tuned.
- **Hibou-B** [ref] (HistAI; ViT-B/14 with 4 register tokens, self-supervised DINOv2 training on
  pathology images; Apache 2.0 licence): tiles resized from 256 to 224 pixels, normalised with
  the encoder's own mean and standard deviation, and represented by the normalised CLS token
  (768 numbers). Main encoder.
- **ResNet-50** [ref] (ImageNet weights, truncated after the third block, global average pooling;
  1,024 numbers): baseline.

Features were computed in half precision on a GPU and stored as float16.

## 4. Slide-level model

Each slide is a bag of tile features with one ISUP label (multiple-instance learning). We used
gated attention MIL [ref Ilse 2018; the single-branch form of CLAM, ref Lu 2021]:
each tile feature is compressed by a linear layer with ReLU and dropout (0.25) to 256 numbers;
a gated attention branch (tanh and sigmoid branches of size 128, multiplied, then a linear
layer) gives one score per tile; a softmax over the tiles turns the scores into attention
weights that sum to 1; the slide embedding is the attention-weighted sum of the compressed tile
features; a linear layer gives 6 class scores (ISUP 0 to 5).

**Training.** Cross-entropy loss, AdamW (learning rate 2 × 10⁻⁴, weight decay 10⁻⁵), cosine
learning-rate decay over 30 epochs, one slide per batch, all tiles of a slide used, training
seed 42. In each fold, the epoch with the highest validation QWK was kept (see the note in
section 11). Training ran on one NVIDIA Tesla T4 (Kaggle), about 2 h for 5 folds.

## 5. Colour handling

The two centres stain and scan differently (Karolinska slides look more purple, Radboud slides
paler pink). Three strategies were compared, each against the same model on the original colours.

**Per-centre feature rescaling.** Each centre's tile features were standardised with that
centre's own mean and standard deviation over all its development tiles (no labels used).

**Stain normalisation (Macenko) [ref Macenko 2009].** For each slide, the stain matrix was
estimated once from 16 evenly spaced tissue tiles (pen ink removed): optical density with
I₀ = 240, pixels with optical density below 0.15 in any channel discarded, the plane of the two
largest eigenvectors, the 1st and 99th percentiles of the angle in that plane as the two stain
vectors, and the 99th percentile of the stain concentrations as the stain strengths. Every tile
was then repainted to a fixed reference stain matrix and reference strengths. Safeguard: when
either estimated stain vector had a cosine similarity below 0.95 to the reference (very pale
slides, where the haematoxylin vector is estimated wrongly), the reference stain vectors were
kept and only the strengths were fitted. This fallback was used on 7.0% of Radboud and 0.9% of
Karolinska development slides.

**Stain augmentation [ref Tellez 2018, 2019].** Each slide was given one random recolouring:
the tile pixels were converted to haematoxylin, eosin and residual amounts with the fixed
Ruifrok–Johnston stain matrix [ref], the haematoxylin and eosin amounts were multiplied by
per-slide factors drawn log-uniformly from [1/1.4, 1.4] and [1/2, 2], and converted back to RGB.
The ranges were chosen from the measured difference between the centres (150 slides per centre:
the median Karolinska slide has 1.15 times the haematoxylin and 2.0 times the eosin of the
median Radboud slide). Recoloured tile features were extracted once; during training each slide
was read with its original or its recoloured features with probability 0.5. Validation and test
slides always kept their original colours. [Results TBD.]

## 6. Generalisation across centres (experiment C)

For each centre and each fold k, one model was trained on that centre's slides outside fold k,
with the epoch chosen on the same centre's fold k. For a test centre B and fold k, the
**within-centre** prediction comes from the model trained on B, and the **cross-centre**
prediction from the model trained on the other centre, both on the same slides of B's fold k.
The two centres share no patients, so the cross-centre test has no patient leakage. Predictions
were pooled over folds; the drop is QWK(cross) − QWK(within) on the same slides, with a paired
bootstrap 95% CI. This design was repeated with ResNet-50 and Hibou-B features, with per-centre
rescaling, with stain normalisation, with both, and with stain augmentation [TBD].

## 7. Patient leakage check

PANDA provides no patient identifiers and several biopsies may come from the same patient, so
slide-level folds may place one patient's biopsies on both sides of a split.

**Step A.** Each slide was summarised by the mean of its Hibou-B tile features, standardised per
centre and scaled to unit length. A slide's twin is the most similar slide (cosine similarity) of
the same centre in the other folds; a close twin is in the top 10% of twin similarity of its
centre (Radboud ≥ 0.960, Karolinska ≥ 0.914). Exact-grade accuracy of the out-of-fold
predictions was compared between slides with and without a close twin within each ISUP grade,
weighted by the close-twin group's grade mix, with a bootstrap 95% CI (1,000 resamples).

**Step B.** Slides of the same centre at or above the close-twin similarity were linked, and
connected groups (single linkage) were kept within one fold (StratifiedGroupKFold, 5 folds,
stratified by centre and grade, seed 42). The Hibou-B model was retrained with these grouped
folds and compared with the original folds on the same slides (paired bootstrap 95% CI). Because
grouping also separates similar-looking slides of different patients, this gives an upper bound
on the effect of leakage.

## 8. Attention analysis

On Radboud ISUP 2 development slides with a label mask, each slide was scored by its own
out-of-fold model. A tile contained Gleason pattern 4 if more than 5% of its pixels were pattern
4 in the mask. Enrichment = share of attention on those tiles ÷ share of tiles that are those
tiles (1 = no preference). Slides without such a tile were skipped. Encoders were compared on
the same slides and tiles (paired). Note: Radboud masks were generated semi-automatically [ref].

## 9. External validation

**PAR** [ref Scientific Data 2026; BioImage Archive S-BIAD2323, CC BY 4.0]: 339 core-needle
biopsy slides from 185 patients (Erbil, Iraq), each scanned on three scanners; slide-level
Gleason and ISUP grades from three pathologists graded independently (pathologist 1: 339 slides,
pathologist 2: 337, pathologist 3: 59). We used the Grundium scans (0.2506 µm/px, 40×).

**PBGG** [ref Tolkach 2023; CC BY-NC-SA 4.0]: 50 biopsy cores, 10 per grade group 1 to 5, each
graded by 10 pathologists (Hamamatsu S360, 0.2305 µm/px). [Grades requested; TBD.]

**Matching the magnification.** External slides were tiled by physical size rather than pyramid
level: tiles covering the same 128 µm as a PANDA tile (target 0.5 µm/px; 511 pixels on a
0.25 µm/px slide) were read at the sharpest level not finer than needed and downsampled to
256 pixels (box filter). The tissue mask was computed at the pyramid level closest to 8 µm/px.
On PANDA slides this procedure gives tile counts within 5% of the original tiling.

**Grading.** External slides were graded by the five PANDA fold models as an ensemble (softmax
probabilities averaged, most likely grade taken). External data were never used for training,
model selection or any setting. Slides were downloaded, processed and deleted one at a time.

## 10. Statistics

- **QWK** (Cohen's kappa with quadratic weights) is the primary metric, as in PANDA.
- **GG1 vs GG2 boundary:** GG2 slides called GG2 (sensitivity for GG2), GG1 slides called GG2,
  GG2 slides called GG1.
- **Confidence intervals:** percentile bootstrap over slides, 2,000 resamples (seed 42) unless
  stated; paired (the same resampled slides for both models) when two models are compared.
- **Paired counts:** exact McNemar test (binomial test on the slides where two models disagree).
- **Decision rules:** before every experiment, a decision rule and a prediction were written
  down and committed (`experiments.md`); results are reported against them, including failures.
- Software: Python, PyTorch, OpenSlide, scikit-learn, scikit-image, Hugging Face transformers.

## 11. Notes for the final version (to resolve, not for the paper as written)

- **Optimistic cross-validation numbers.** The epoch kept in each fold is the one with the best
  validation QWK, and the reported QWK is measured on that same validation fold, so the
  cross-validation QWK is slightly optimistic. All comparisons share this bias, so differences
  between models are fair, but the absolute number must come from the locked test set (or from
  a fixed number of epochs / an inner validation split). Decide before unlocking the test set.
  Measured 2026-11-26: best-epoch vs last-epoch validation QWK differ by 0.003 to 0.007 (Hibou-B
  0.884 vs 0.877), so the bias is small and the comparisons hold.
- **One training seed per experiment.** The seed check (2026-11-26) will show how much results
  move between identical trainings. [TBD]
- **Patients.** Similarity is only a stand-in for patient identity.
- **Two centres in development.** External validation (PAR, PBGG) addresses this. [TBD]
- Citations still to add: PANDA, Hibou-B, ResNet, Ilse 2018, CLAM, Macenko, Ruifrok–Johnston,
  Tellez, PAR, PBGG (Tolkach 2023), the Radboud mask paper.
