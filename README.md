# GreenBridge

Can a satellite-image classifier trained in the Netherlands generalise to India?

A small, reproducible transfer-learning project comparing Rotterdam and Bengaluru. It trains a three-class classification head on frozen pretrained ResNet-18 features, then compares Dutch-only training with joint Dutch/Indian training on the same held-out test sets. This is an initial baseline, not full-network fine-tuning.

**Status:** training pipeline ready; real local imagery and labels still need collecting. No accuracy or social-impact claims have been measured. The example manifest shows the format, not a usable dataset.

## Start in Google Colab (recommended)

Upload this project folder and your labelled data to a Colab session, change into its directory, and run:

```python
!pip install -r requirements.txt
!python train.py --manifest data/manifest.csv --epochs 5
```

The first run downloads pretrained weights. Training uses the CPU for an accessible baseline. Save the runs folder before your Colab session ends.

For local Windows use, create a Python 3.12 environment, then install requirements and run the same training command. The machine this project was prepared on has Python 3.15, outside the currently documented PyTorch Windows support range (3.10–3.14): https://pytorch.org/get-started/locally/

## Collect your first dataset

1. Open https://dataspace.copernicus.eu/browser/ and choose Sentinel-2 L2A true-colour imagery for Rotterdam and Bengaluru. Follow the source's access and attribution terms.
2. Use cloud-free imagery, the same ground resolution, RGB rendering and patch ground footprint in both cities. A suggested start is 64 × 64 pixels at 10 m resolution. Keep scene IDs, acquisition dates, bounding boxes, processing settings and attribution in a separate provenance file.
3. Collect patches dominated by vegetation, built-up surfaces, or water. Skip cloudy and ambiguous mixed patches. Target 30 training and 15 test examples per class per country for a pilot; this is too small for strong generalisation claims.
4. Reserve geographically separate areas for testing BEFORE extracting patches. Never allow overlapping patches or alternate dates of the same location across train and test. Use a buffer between areas. A group is a spatial area, not an individual image or acquisition date.
5. Save images under data/images/. Copy manifest.example.csv to data/manifest.csv and replace the example rows. Paths are relative to the manifest. Country is NL or IN; labels are vegetation, built_up, water; split is train or test. Include every class in each country/split.

The loader checks exact duplicate image content and spatial-group leakage. It cannot detect nearby or overlapping locations: you must enforce geographical separation while collecting data. Hold the test sets fixed and do not choose settings using test results. Add a separate geographic validation set before tuning hyperparameters.

## What you get

- runs/nl_only.pt and runs/nl_plus_india.pt: trained model checkpoints.
- runs/results.json: accuracy, macro F1 and confusion matrices for both countries, plus seed and manifest hash.
- Matrix rows are actual classes; columns are predictions, in the reported class order.

Joint training starts from the identical initial model as Dutch-only training. The larger joint dataset means more optimizer steps per epoch: improvement cannot be attributed solely to cross-country diversity. A later experiment should add a matched-sample or matched-update control and repeat across seeds.

## Honest portfolio scope

This detects dominant land cover in patches. It does not measure park access, tree canopy precisely, heat exposure, or where a park should be built. Two-city findings do not represent either whole country. Seasonal differences, image rendering and label quality may explain performance changes.

After the first real run: inspect errors, repeat seeds, document dataset provenance, and add a map of held-out patch predictions. Publish measured results and limitations alongside the demo. Do not upload third-party imagery without checking redistribution rights.

## References

- Sentinel imagery access: https://documentation.dataspace.copernicus.eu/Applications/Browser.html
- Optional introductory dataset (not a substitute for local labels): https://arxiv.org/abs/1709.00029
- ResNet-18: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
