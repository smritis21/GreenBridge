# GreenBridge

Train in the Netherlands. Test in India. Measure what changes.

GreenBridge is a terminal-based satellite image classification project for Rotterdam and Bengaluru. It downloads small Sentinel-2 true-colour crops, creates spatially separated patches, helps you review labels with Claude, trains two local classifiers, and writes measured results.

## What uses Claude?

Claude suggests image labels and can explain your final metrics. You enter your Anthropic API key in a hidden terminal prompt. It is not written to disk. You can also set ANTHROPIC_API_KEY yourself. Your image patches are sent to Anthropic only when you use `label --claude`; `report --claude` sends the metrics JSON. Each call uses your API credits. Limits and cached suggestions help control spending.

The classifier you train is a pretrained ResNet-18 with a new three-class head. Claude API credits pay for Claude calls, not local PyTorch training, and the project does not fine-tune Claude. Running local training needs no API key. No training or paid API calls were run during development.

## Windows terminal setup

Use Python 3.12. If it is not installed, run the following, then open a new terminal:

```powershell
winget install --exact --id Python.Python.3.12
```

For a fresh download:

```powershell
git clone https://github.com/smritis21/GreenBridge.git
cd GreenBridge
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you already have this folder open in VS Code, run the setup commands starting at `py -3.12`. There is no need to activate the environment or change PowerShell execution policy. If your existing .venv uses Python 3.15, create `.venv312` instead and use that name in every command below.

## 1. Download satellite imagery

```powershell
.\.venv\Scripts\python.exe greenbridge.py fetch
```

The public Planetary Computer service supplies small RGB crops for both cities. No satellite API key is required. The default searches 2024 for scenes with less than 10 percent scene-wide cloud cover. That does not guarantee a cloud-free local crop. Check every patch. Source scene metadata is saved next to each image. Existing downloads with metadata are kept when you rerun fetch.

If no suitable scene covers an area, try another interval and output folder:

```powershell
.\.venv\Scripts\python.exe greenbridge.py fetch --dates "2023-01-01T00:00:00Z/2023-12-31T23:59:59Z" --output data/source2023
```

## 2. Prepare patches

```powershell
.\.venv\Scripts\python.exe greenbridge.py prepare
```

Default: 64 by 64 pixels, at most 90 patches per country per split. Training uses the left side of each image and testing the right side, with a two-patch gap. The manifest contains blank labels. The command refuses to replace an existing patch folder.

For alternate source images use `--nl data/source2023/NL.png --india data/source2023/IN.png`. For a larger sample use `--limit 300 --output data/patches_large`, then pass that manifest to label, check and train.

## 3. Label with Claude help

```powershell
.\.venv\Scripts\python.exe greenbridge.py label --claude --limit 12
```

Enter your API key when prompted. An image preview opens and Claude suggests a class. Type your verified label: `v` for vegetation, `b` for built-up land, `w` for water, `s` for skip, or `q` to quit. Repeat the command to continue. Previously reviewed labels are kept and Claude suggestions are cached separately.

Use `--no-preview` if you want to inspect image files yourself. Use `--model YOUR_MODEL_ID` or the CLAUDE_MODEL environment variable to select a model available in your Anthropic account. The default is `auto`, which selects the first available Sonnet from the API model list. Run `python greenbridge.py models` to list model IDs available to your key. An unavailable model or insufficient API credit will cause the API to reject the request.

For manual labelling without any API calls:

```powershell
.\.venv\Scripts\python.exe greenbridge.py label --limit 30
```

Skip clouds, blank pixels, and ambiguous mixed patches. Confirm labels yourself. Claude suggestions are not ground truth. For a stronger study, label the test set independently without viewing Claude suggestions and have another person review it.

## 4. Check your dataset

```powershell
.\.venv\Scripts\python.exe greenbridge.py check
```

Every country needs all three confirmed classes in both train and test. Aim for at least 30 training and 15 test patches per class per country for an initial pilot. That is still a small sample. Blank and skipped labels are excluded. Checks reject duplicate file content and spatial groups that cross split boundaries. They do not detect all geographic overlaps or validate whether your labels are correct.

If a class is missing, label more patches. If the crop lacks that class, collect a larger area or use custom source crops and a new manifest. Do not relabel images just to fill a class.

## 5. Train both models

```powershell
.\.venv\Scripts\python.exe train.py --epochs 5
```

The first run downloads pretrained weights. Training runs on CPU. It compares Dutch-only training against joint Dutch and Indian training, starting from identical initial weights. Outputs: `runs/nl_only.pt`, `runs/nl_plus_india.pt`, and `runs/results.json`.

For a repeat with another seed, use a new output folder:

```powershell
.\.venv\Scripts\python.exe train.py --epochs 5 --seed 43 --output runs_seed43
```

Accuracy, macro F1 and confusion matrices are reported separately for NL and IN. Confusion-matrix rows are actual labels and columns are predictions. Class order is included in results.json. Results also record the manifest hash and run settings.

## 6. Predict one image

Replace the image path with one of your actual patches:

```powershell
.\.venv\Scripts\python.exe greenbridge.py predict --image data/patches/images/NL_0_0.png
```

The output gives a predicted label and class scores. Scores are not calibrated confidence. Use only checkpoints you created or trust.

## 7. Create your report

```powershell
.\.venv\Scripts\python.exe greenbridge.py report
```

Optional Claude commentary:

```powershell
.\.venv\Scripts\python.exe greenbridge.py report --claude
```

The report is saved to `runs/report.md`. Claude commentary is explicitly labelled as AI generated. A plain report needs no API key.

## Checks without running models

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

The tests cover spatial split preparation, duplicate detection, missing classes, unreviewed labels, Claude suggestion separation and measured report values. API calls are mocked. GitHub Actions runs these tests without API secrets or model training.

## Scope and limits

This is a runnable research prototype, not a validated planning tool. No trained weights, real labelled dataset or benchmark scores are supplied. Raw imagery, keys and model files are excluded from Git.

The model detects dominant land cover in image patches. It does not measure park accessibility, precise canopy cover, heat exposure or where new parks should be built. Two city samples cannot represent whole countries. Seasonal differences, RGB rendering, label quality and spatial sampling can affect results.

The joint dataset gives more optimiser steps per epoch than the Dutch-only dataset. Any improvement could reflect extra data or training, rather than cross-country diversity alone. Add matched-sample controls, a separate geographic validation set and multiple seeds before making strong claims. Do not tune settings against test results.

Source images use geographic coordinates, but the first version produces patch predictions rather than a geographic map. The project includes no web app.

## Data and API references

- [Planetary Computer public data access](https://planetarycomputer.microsoft.com/docs/)
- [Signing satellite assets](https://planetarycomputer.microsoft.com/docs/concepts/sas/)
- [Claude Python SDK](https://platform.claude.com/docs/en/api/sdks/python)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [EuroSAT background paper](https://arxiv.org/abs/1709.00029)

Keep scene attribution and consult source data terms before redistributing imagery. Software dependencies and satellite data retain their own licences.
