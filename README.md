<p align="center"><img src="assets/banner.svg" alt="GreenBridge: Train in the Netherlands. Test in India." width="100%"></p>

<p align="center">
<img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-86BDCC?style=flat-square">
<img alt="PyTorch" src="https://img.shields.io/badge/ML-PyTorch-F4B780?style=flat-square">
<img alt="Claude-assisted labels" src="https://img.shields.io/badge/Labels-Claude_assisted-C0ED90?style=flat-square">
<img alt="Status: preliminary prototype" src="https://img.shields.io/badge/Status-Preliminary_prototype-D7C79E?style=flat-square">
</p>

<p align="center"><strong>A small experiment in satellite imagery, transfer learning, and what happens when data crosses borders.</strong></p>
<p align="center"><a href="https://smritis21.github.io/GreenBridge/">GitHub Pages website</a> · <a href="#website-preview">Website screenshots</a> · <a href="docs/RUNNING.md">Full terminal guide</a> · <a href="results/experiment-01.md">Recorded results</a></p>

## The question

Can a land-cover classifier trained around Rotterdam recognise vegetation, built-up land and water around Bengaluru? Does adding Indian training imagery improve its results?

GreenBridge provides a terminal workflow to collect imagery, prepare spatially separated patches, label training data with Claude, train two classifiers and inspect their mistakes. A separate rural area near Bengaluru supplements the urban Indian training crop.

> **Current status:** the pipeline runs and the first experiment is recorded. The evaluation is small and incomplete, especially for India. This is an exploratory prototype, not a validated environmental mapping tool.

## Website preview

The project website explains the question, method, first results, limitations and next steps. Its green, orange and blue palette matches vegetation, built-up land and water. The hero illustration is a schematic, not satellite data or a model prediction.

[Open the local website](http://localhost:8766/) while its preview server is running. Visit the live website: **[smritis21.github.io/GreenBridge](https://smritis21.github.io/GreenBridge/)**. Its entry point is [index.html](index.html) in the repository root.

### Homepage

![GreenBridge website homepage with green typography and an illustrated landscape](assets/website-home.jpg)

### Results

![GreenBridge website results showing 76 percent versus 68 percent Dutch accuracy and the limited Indian test coverage](assets/website-results.jpg)

## How it works

| Stage | What happens |
|---|---|
| Collect | Download Sentinel-2 RGB crops through Microsoft Planetary Computer. |
| Prepare | Create 64 × 64 patches with separate training and test areas. |
| Label | Claude labels training images; humans label test images. Label sources are recorded. |
| Train | Train a new classification head on frozen pretrained ResNet-18 features. |
| Compare | Evaluate Dutch-only and joint Dutch/Indian training on the same test patches. |
| Report | Save model checkpoints, confusion matrices, class coverage and a readable report. |

**Colour key:** 🟢 vegetation · 🟠 built-up land · 🔵 water

Claude supplies training labels and optional explanations. It is not the model being trained. Its API calls use Anthropic credits; local classifier training does not.

## Dataset source and preparation

This is a **custom patch dataset built from public satellite imagery**, not a downloaded labelled benchmark. EuroSAT was mentioned as a possible learning resource but was not used in the recorded experiment.

**Original imagery:** Copernicus Sentinel-2 Level-2A, accessed through the [Microsoft Planetary Computer Sentinel-2 L2A collection](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a). The code searches the public STAC API and downloads small windows from its true-colour visual assets. These are 8-bit RGB images at 10 metres per pixel, not all Sentinel spectral bands.

| Area | Acquisition time (UTC) | Crop bounds: west, south, east, north | Use |
|---|---|---|---|
| Rotterdam, Netherlands | 2024-08-12T10:46:29.024000Z | 4.4, 51.88, 4.55, 51.98 | Training and test |
| Bengaluru, India | 2024-03-13T05:06:49.024000Z | 77.52, 12.9, 77.67, 13.0 | Training and test |
| Rural area west of Bengaluru, India | 2024-03-13T05:06:49.024000Z | 77.25, 12.8, 77.4, 12.9 | Training only |

### Exact scenes

- **Rotterdam, Netherlands:** `S2B_MSIL2A_20240812T104629_R051_T31UET_20240812T134520`. Coordinate system: `EPSG:32631`.
- **Bengaluru, India:** `S2B_MSIL2A_20240313T050649_R019_T43PGQ_20240313T171334`. Coordinate system: `EPSG:32643`.
- **Rural area west of Bengaluru, India:** `S2B_MSIL2A_20240313T050649_R019_T43PGQ_20240313T171334`. Coordinate system: `EPSG:32643`.

The two Indian crops come from the same satellite scene, but different geographic windows. The Netherlands and India acquisitions are in different months, which can introduce seasonal differences.

### How the images became training data

- The download searches the 2024 interval for scenes with less than 10% scene-wide cloud cover. This does not guarantee cloud-free patches.
- Images are divided into non-overlapping 64 × 64 pixel patches, roughly 640 × 640 metres per patch at the source resolution.
- Original city crops use their left half for training and right half for testing, with a two-patch gap. The separate rural Indian crop is training only and does not overlap the original Indian crop's bounding box.
- The first preparation took the top rows first. Later repair samples unused training patches in shuffled order. New preparation runs shuffle across the full crop. The recorded experiment therefore reflects an evolving, non-random overall sample.
- The three classes are vegetation, built-up land and water. Cloudy, blank or mixed/uncertain patches can be marked skip. Skipped and unreviewed rows are excluded.
- Most training labels were generated using the Claude API. The labelling session selected `claude-sonnet-5-5`; per-image model names are stored locally where recorded. Earlier cached suggestions may use other recorded models.
- Human labels were entered manually. Some human reviewers saw Claude suggestions, so this is not an independently annotated reference dataset. No expert ground-truth validation was performed.
- The saved experiment records 201 Claude-labelled usable patches, 18 explicitly human-labelled usable patches and 23 older usable patches with a blank source field, across training and testing combined. These totals are not a training-only breakdown.
- The joint run used 214 training patches; the Dutch-only run used 83. The fixed test set had 25 Dutch patches (21 vegetation, 3 built-up, 1 water) and 3 Indian patches (all built-up).
- The pretrained ResNet-18 transform resizes, centre-crops and normalises RGB patches before the classifier sees them. The backbone is frozen and only the new classification head is trained. No Claude model is trained or fine-tuned.

### Provenance, attribution and redistribution

[Full source metadata](results/dataset-provenance.json) includes the exact scene identifiers, acquisition times, bounding boxes, source asset URLs, coordinate systems and crop transforms. These are unsigned public asset URLs, not API credentials. [Experiment JSON](results/experiment-01.json) records the seed, epoch count, class support and manifest hash. Raw imagery and trained weights are not uploaded to this repository. The local manifest is not included, so the published metadata alone is not enough to reproduce the exact selected patches and labels.

Attribution: **Contains modified Copernicus Sentinel data (2024), accessed via Microsoft Planetary Computer.** Consult the [Sentinel data legal notice](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice) and the collection terms before redistributing data. This project makes no claim of endorsement by Copernicus, ESA, Microsoft or Anthropic.

## First experiment

Seed 42, five epochs per model. Dutch-only training used 83 images; joint training used 214.

| Model | Test area | Images | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| Netherlands only | Netherlands | 25 | 76% | 0.463 |
| Netherlands + India | Netherlands | 25 | 68% | 0.384 |
| Netherlands only | India | 3 | 3/3 correct | 0.333 |
| Netherlands + India | India | 3 | 3/3 correct | 0.333 |

**What happened:** adding Indian training data did not improve Dutch test performance in this run. The drop does not establish a cause. Both models missed the single Dutch water test image.

**Why the Indian result is limited:** all three Indian test examples are built-up. There are no Indian test examples for water or vegetation. These results check that the pipeline runs; they cannot establish generalisation across Indian land-cover classes. Macro F1 averages all three classes, with absent classes contributing zero.

[Raw experiment JSON](results/experiment-01.json) · [Generated report](results/experiment-01.md)

## Quick start

Use Python 3.12. In PowerShell, from the repository folder:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe greenbridge.py --help
```

The full download, labelling, training and prediction commands are in the [terminal guide](docs/RUNNING.md). Enter the Anthropic API key at the hidden prompt when using Claude. Do not put it in source files.

### View the website locally

```powershell
.\.venv312\Scripts\python.exe -m http.server 8766 --bind 127.0.0.1 --directory .
```

Open [localhost:8766](http://localhost:8766/). The website is static HTML and CSS and needs no API key or build step. Stop the server with Ctrl+C. The design uses Google Fonts when available and system fonts otherwise.

### Host with GitHub Pages

In the repository settings, open **Pages**, select **Deploy from a branch**, then choose **main** and **/ (root)**. Save the setting. The live website is [smritis21.github.io/GreenBridge](https://smritis21.github.io/GreenBridge/).

The site uses relative asset paths, so it works under the GreenBridge repository URL. No build step or API key is needed.

### Run checks without model training

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -v
```

Sixteen tests currently cover spatial separation, duplicate detection, label handling, Claude response retries, automatic training labels and limited test coverage. Paid calls are mocked in tests. GitHub Actions also runs the checks.

## Limits worth keeping visible

- Training labels are mostly AI generated and may be wrong. Older labels have incomplete source metadata.
- The test set is small and uneven. Some labels were reviewed with Claude suggestions visible, so it is not an independently annotated benchmark.
- The joint dataset produces more training updates per epoch. More data and more updates are confounded.
- Geography, season, image rendering and sampling can affect performance. Only one seed has been run.
- Training includes a nearby rural Indian area, while the current Indian test uses urban images.
- This classifies dominant cover in patches. It does not measure park access, heat exposure or where new parks should go.

## Next steps

- [ ] Verify suitable existing labelled datasets covering both countries.
- [ ] Build larger, independently labelled and geographically separate test sets.
- [ ] Add a simple baseline and matched-data comparisons.
- [ ] Repeat across seeds and show variation.
- [ ] Visualise predictions and errors on a map.

## Repository map

```text
greenbridge.py       Terminal workflow and Claude integration
train.py             Two-model training and evaluation
test_greenbridge.py  Checks without paid API calls
index.html           GitHub Pages entry point
style.css            Website styles
landscape.svg        Website illustration
assets/              README banner and actual website screenshots
results/             Recorded first experiment, without model weights
docs/RUNNING.md      Detailed terminal instructions
```

Raw images, trained model weights, API keys and local environments are not committed. Satellite data and dependencies retain their own terms. Website illustration and banner are schematic project artwork.

Built by [Smriti Sridhar](https://github.com/smritis21).
