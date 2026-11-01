"""Compare Netherlands-only training with joint Netherlands/India training."""
import argparse
import copy
import csv
import hashlib
import json
import random
import warnings
from pathlib import Path

CLASSES = ['vegetation', 'built_up', 'water']


def read_manifest(path):
    if not path.exists():
        raise FileNotFoundError(f'Manifest not found: {path}')

    with path.open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Manifest is empty: {path}')

    required = {'path', 'country', 'label', 'split', 'group'}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f'Manifest missing required columns: {sorted(missing)}')

    # Unreviewed and explicitly skipped images never enter training or evaluation.
    rows = [r for r in rows if r['label'] not in ('', 'skip')]
    seen, groups = set(), {}
    for row in rows:
        if row['country'] not in ('NL', 'IN') or row['split'] not in ('train', 'test') or row['label'] not in CLASSES:
            raise ValueError('Invalid country, split, or label')
        if row['split'] == 'test' and row.get('label_source') == 'claude':
            raise ValueError('Test labels must be human reviewed')
        if not row.get('path'):
            raise ValueError('Manifest row is missing an image path')

        if not row.get('group', '').strip():
            raise ValueError('Every image needs a spatial group')
        image = (path.parent / row['path']).resolve()
        if not image.is_file():
            raise FileNotFoundError(f'Image not found: {image}')

        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError('Duplicate image content: ' + str(image))
        seen.add(digest)
        group = (row['country'], row['group'])
        if group in groups and groups[group] != row['split']:
            raise ValueError('Spatial group crosses train/test boundary')
        groups[group] = row['split']
        row['image'] = image
    for country in ('NL', 'IN'):
        for split in ('train', 'test'):
            present = {r['label'] for r in rows if r['country'] == country and r['split'] == split}
            if split == 'train' and present != set(CLASSES):
                raise ValueError(f'Need all three classes in {country}/{split}')
            if split == 'test' and not present:
                raise ValueError(f'Need at least one human-labelled image in {country}/test')
            if split == 'test' and present != set(CLASSES):
                warnings.warn(f'{country}/test is missing {sorted(set(CLASSES) - present)}. '
                              'This only checks the pipeline, not performance across all classes.')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path('data/patches/manifest.csv'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('runs'))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    rows = read_manifest(args.manifest)
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models import resnet18, ResNet18_Weights

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    class Patches(Dataset):
        def __init__(self, records):
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, index):
            row = self.records[index]
            with Image.open(row['image']) as im:
                x = transform(im.convert('RGB'))
            return x, CLASSES.index(row['label'])

    base = resnet18(weights=weights)
    for p in base.parameters():
        p.requires_grad = False
    base.fc = torch.nn.Linear(base.fc.in_features, len(CLASSES))
    if (args.output / 'results.json').exists():
        parser.error('Results already exist. Choose a new --output folder.')
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, countries in [('nl_only', ['NL']), ('nl_plus_india', ['NL', 'IN'])]:
        model = copy.deepcopy(base)
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
        generator = torch.Generator().manual_seed(args.seed)
        train = [r for r in rows if r['split'] == 'train' and r['country'] in countries]
        loader = DataLoader(Patches(train), batch_size=16, shuffle=True, generator=generator)
        for epoch in range(args.epochs):
            model.eval()  # Preserve pretrained batch-normalization statistics.
            model.fc.train()
            total = 0.0
            for x, y in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y)
            print(f'{name}: epoch {epoch + 1}, loss {total / len(train):.4f}', flush=True)
        model.eval()
        results[name] = {'training_samples': len(train), 'evaluation': {}}
        for country in ('NL', 'IN'):
            test = [r for r in rows if r['split'] == 'test' and r['country'] == country]
            matrix = torch.zeros((3, 3), dtype=torch.int64)
            with torch.no_grad():
                for x, y in DataLoader(Patches(test), batch_size=16):
                    predictions = model(x).argmax(1)
                    for actual, predicted in zip(y, predictions):
                        matrix[actual, predicted] += 1
            f1 = [2 * matrix[i, i].item() / max(1, (matrix[i, :].sum() + matrix[:, i].sum()).item()) for i in range(3)]
            results[name]['evaluation'][country] = {
                'class_support': {CLASSES[i]: matrix[i, :].sum().item() for i in range(3)},
                'missing_test_classes': [CLASSES[i] for i in range(3) if matrix[i, :].sum().item() == 0],
                'samples': len(test), 'accuracy': matrix.diag().sum().item() / len(test),
                'macro_f1': sum(f1) / 3, 'confusion_matrix': matrix.tolist()}
        torch.save({'state_dict': model.state_dict(), 'classes': CLASSES}, args.output / f'{name}.pt')
    report = {'seed': args.seed, 'epochs': args.epochs, 'classes': CLASSES,
              'label_sources': {source: sum(r.get('label_source', 'human_legacy') == source for r in rows)
                                for source in {r.get('label_source', 'human_legacy') for r in rows}},
              'label_note': 'Training may use AI-generated labels. Test labels are human reviewed.',
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(), 'results': results}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
