# External validation: candidate datasets

Everything Iris has been tested on so far comes from PANDA (Radboud and Karolinska). An external
test needs prostate **biopsies** with **slide-level ISUP/Gleason grades** from a **different
centre**, under a licence that allows research publication. Searched 2026-11-26.

| Dataset | Origin | Content | Labels | Access and licence |
|---|---|---|---|---|
| **PAR** ([Scientific Data 2026](https://www.nature.com/articles/s41597-026-07798-9)) | Erbil, Iraq | 339 biopsy slides, 185 patients, each scanned on 3 scanners (Grundium, Hamamatsu, Leica) = 1,017 WSIs | Gleason + ISUP from 3 pathologists, independent | open, CC BY 4.0, [BioImage Archive S-BIAD2323](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2323) |
| **SICAPv2** | Valencia, Spain | 155 biopsies, 95 patients | slide Gleason score + patch grades | open ([Mendeley Data](https://data.mendeley.com/datasets/9xxm58dvs3/2)); licence to check |
| **PBGG-1/2** ([Tolkach 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10427608/)) | international | 50 biopsy cores | 10 pathologists | images open ([part 1](https://zenodo.org/record/8102833), [part 2](https://zenodo.org/record/8102929)); grades by email request; CC BY-NC-SA |
| **SPROB20** | Sweden | 2,611 biopsies, 460 cases | Gleason/ISUP from clinical work | controlled access, data sharing agreement ([AIDA](https://datahub.aida.scilifelab.se/10.23698/aida/sprob20)) |
| **AGGC 2022** | Singapore | 53 biopsies + 144 prostatectomies | Gleason, pixel annotations | registration; CC BY-NC-SA ([site](https://aggc22.grand-challenge.org)) |
| **Tan Tock Seng** | Singapore | 59 biopsies + 40 prostatectomies | Gleason score | open ([Zenodo](https://zenodo.org/record/7152243)) |
| BRIDGE | Poland | 869 biopsy slides, 169 patients | ISUP (software-assisted) | licence not found yet ([arXiv](https://arxiv.org/html/2611.13883v1)) |
| TCGA-PRAD | USA | ~500 slides | Gleason | open, but prostatectomies, not biopsies |

**Plan:** PAR as the main external test (generalisation to a new centre and population, and
scanner robustness); SICAPv2 as a second centre; PBGG for the GG1 vs GG2 boundary against the
spread between 10 pathologists.

## PAR in detail

Files: `https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/323/S-BIAD2323/Files/bia_prostate_wsi/`

| Folder | Scanner | Size |
|---|---|---|
| `component_A_grundium_svs` | Grundium Ocus40, .svs, 40× (0.2506 µm/px) | 114 GB |
| `component_B_hamamatsu_ndpi` | Hamamatsu NanoZoomer, .ndpi | 232 GB |
| `component_C_leica_svs` | Leica Aperio GT 450, .svs | 249 GB |
| `annotations/slide_labels.tsv` | grades per slide (`slide_id`, `gleason_r1..3`, `isup_1`, `isup_r2`, `isup_r3`) | 8 KB |

Too big to store on the laptop, so slides are **streamed**: download one, tile, extract
features, delete (`scripts/13_external_features.py`), on Kaggle (internet + GPU).

ISUP grades (pathologist 1 graded all 339 slides, pathologist 2 graded 337, pathologist 3 graded 59):

| ISUP | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Pathologist 1 | 164 | 7 | 38 | 59 | 30 | 41 |
| Pathologist 2 | 209 | 2 | 7 | 8 | 25 | 86 |

- **Almost no GG1 slides** (7 or 2): PAR cannot test the GG1 vs GG2 boundary, only overall grading,
  a new centre and scanners.
- **The pathologists disagree a lot:** QWK 0.811 between pathologists 1 and 2, the same exact grade
  on 64% of slides, cancer vs benign the same on 86%. Pathologist 2 calls 20 of pathologist 1's
  GG2 slides benign. This is the human benchmark for Iris.

## Matching PANDA's zoom

PANDA 20× tiles are 256 px at level 0: 0.486 µm/px (Radboud), 0.503 µm/px (Karolinska), about
128 × 128 µm of tissue. `MicronTiler` (`iris/data/tiling.py`) reads each slide's µm/px and cuts
squares covering 128 µm (512 px on a 0.25 µm/px slide), shrunk to 256 px. Checked 2026-11-26:
on PANDA slides it gives tile counts within 5% of the original 20× tiling, and 4 of 4 PANDA
slides are graded right through the new code path (benign ×2, GG2, GG4).

## First look (3 slides, not a result)

Three PAR slides, Grundium scans, each with the same grade from all 3 pathologists. Original
colours = Hibou-B ensemble on the original features; stain = Macenko features graded by the
stain-normalised model.

| Slide | Pathologists | Original colours | Stain-normalised |
|---|---|---|---|
| C038A | benign | **ISUP 4** (97% of the probability on ISUP 3 to 5) | ISUP 1 |
| C040A | GG2 | ISUP 3 | ISUP 4 |
| C002A | GG5 | ISUP 5 (99.6%) | ISUP 5 |

- GG5 is graded right by both; the benign and GG2 slides are graded too high, like Radboud →
  Karolinska (strongly coloured slides called high grade).
- Macenko's fallback to the reference colours was used on **all three** PAR slides: their stain
  colours are too far from normal H&E for Macenko to estimate, so normalisation only rescales the
  stain strengths there. This may limit what stain normalisation can do on PAR, and makes stain
  augmentation worth testing on it.

Three slides only; the real test needs all 339 slides and a decision rule written first.
