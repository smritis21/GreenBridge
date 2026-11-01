import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import greenbridge as gb
from train import read_manifest


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self):
        rows = []
        for country in ('NL', 'IN'):
            for split in ('train', 'test'):
                for label in gb.CLASSES:
                    name = f'{country}_{split}_{label}.png'
                    (self.root / name).write_bytes(name.encode())
                    rows.append(dict(path=name, country=country, split=split,
                                     label=label, group=f'{country}-{split}'))
        path = self.root / 'manifest.csv'
        gb.write_csv(path, rows)
        return path, rows

    def test_valid_manifest_and_unreviewed_rows(self):
        path, rows = self.manifest()
        rows.append(dict(path='not_downloaded.png', country='NL', label='', split='train', group='x'))
        gb.write_csv(path, rows)
        self.assertEqual(len(read_manifest(path)), 12)

    def test_duplicate_content_is_rejected(self):
        path, rows = self.manifest()
        (self.root / rows[1]['path']).write_bytes((self.root / rows[0]['path']).read_bytes())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_manifest(path)

    def test_spatial_leakage_is_rejected(self):
        path, rows = self.manifest()
        rows[3]['group'] = rows[0]['group']
        gb.write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, 'Spatial'):
            read_manifest(path)

    def test_missing_class_is_rejected(self):
        path, rows = self.manifest()
        gb.write_csv(path, rows[:-1])
        with self.assertRaisesRegex(ValueError, 'all three'):
            read_manifest(path)

    def test_prepare_creates_separated_patches(self):
        from PIL import Image
        source = self.root / 'source.png'
        Image.new('RGB', (640, 192), 'green').save(source)
        dest = self.root / 'patches'
        gb.prepare(argparse.Namespace(nl=source, india=source, output=dest, size=64, limit=9))
        with (dest / 'manifest.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        train_x = [int(r['path'].split('_')[1]) for r in rows if r['split'] == 'train']
        test_x = [int(r['path'].split('_')[1]) for r in rows if r['split'] == 'test']
        self.assertGreaterEqual(min(test_x) - max(train_x), 192)
        self.assertTrue(all(r['label'] == '' for r in rows))

    def test_claude_suggestion_does_not_become_ground_truth(self):
        from PIL import Image
        image = self.root / 'patch.png'
        Image.new('RGB', (64, 64), 'green').save(image)
        manifest = self.root / 'manifest.csv'
        gb.write_csv(manifest, [dict(path='patch.png', country='NL', label='', split='test', group='east')])
        args = argparse.Namespace(manifest=manifest, claude=True, model='test', limit=1, no_preview=True)
        with patch.object(gb, 'claude_client', return_value=object()), patch.object(
                gb, 'ask_image', return_value={'label': 'vegetation', 'reason': 'green'}), patch('builtins.input', return_value='q'):
            gb.label(args)
        with manifest.open() as stream:
            self.assertEqual(next(csv.DictReader(stream))['label'], '')
        self.assertTrue((self.root / 'claude_suggestions.json').exists())

    def test_incomplete_claude_reply_is_retried(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        image = self.root / 'image.png'
        image.write_bytes(b'image')
        create = Mock(side_effect=[
            SimpleNamespace(content=[SimpleNamespace(type='text', text='{"label":')], stop_reason='max_tokens'),
            SimpleNamespace(content=[SimpleNamespace(type='text', text='{"label":"water","reason":"A lake"}')], stop_reason='end_turn')])
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        self.assertEqual(gb.ask_image(client, 'test', image)['label'], 'water')
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args.kwargs['max_tokens'], 2048)

    def test_invalid_claude_replies_stop_after_two_attempts(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        image = self.root / 'image.png'
        image.write_bytes(b'image')
        create = Mock(return_value=SimpleNamespace(content=[SimpleNamespace(type='text', text='broken')], stop_reason='end_turn'))
        with self.assertRaisesRegex(ValueError, 'Previous labels are saved'):
            gb.ask_image(SimpleNamespace(messages=SimpleNamespace(create=create)), 'test', image)
        self.assertEqual(create.call_count, 2)

    def test_auto_model_uses_available_sonnet(self):
        from types import SimpleNamespace
        client = SimpleNamespace(models=SimpleNamespace(list=lambda **kw: [
            SimpleNamespace(id='claude-haiku-example'), SimpleNamespace(id='claude-sonnet-example')]))
        self.assertEqual(gb.resolve_model(client, 'auto'), 'claude-sonnet-example')
        self.assertEqual(gb.resolve_model(client, 'explicit-id'), 'explicit-id')

    def test_auto_model_rejects_no_sonnet(self):
        from types import SimpleNamespace
        client = SimpleNamespace(models=SimpleNamespace(list=lambda **kw: []))
        with self.assertRaisesRegex(ValueError, 'No Sonnet'):
            gb.resolve_model(client, 'auto')

    def test_auto_labels_only_blank_training_rows(self):
        path, rows = self.manifest()
        rows[0]['label'] = ''
        rows[3]['label'] = ''
        gb.write_csv(path, rows)
        args = argparse.Namespace(manifest=path, limit=180, model='test')
        with patch.object(gb, 'claude_client', return_value=object()), patch.object(
                gb, 'ask_image', return_value={'label': 'vegetation', 'reason': 'green'}) as ask:
            gb.auto_label(args)
        with path.open() as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(saved[0]['label_source'], 'claude')
        self.assertEqual(saved[3]['label'], '')
        self.assertEqual(saved[1]['label'], rows[1]['label'])
        with patch.object(gb, 'claude_client') as client:
            gb.auto_label(args)
            client.assert_not_called()

    def test_ai_test_labels_are_rejected(self):
        path, rows = self.manifest()
        rows[3]['label_source'] = 'claude'
        gb.write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, 'human reviewed'):
            read_manifest(path)

    def test_repair_preserves_labels_and_test_set(self):
        from PIL import Image
        source = self.root / 'source.png'
        Image.new('RGB', (640, 320), 'green').save(source)
        dest = self.root / 'patches'
        gb.prepare(argparse.Namespace(nl=source, india=source, output=dest, size=64, limit=2))
        manifest = dest / 'manifest.csv'
        with manifest.open() as stream:
            before = list(csv.DictReader(stream))
        before[0]['label'] = 'built_up'
        gb.write_csv(manifest, before)
        args = argparse.Namespace(manifest=manifest, limit=1, target=5, model='test')
        with patch.object(gb, 'claude_client', return_value=object()), patch.object(
                gb, 'ask_image', return_value={'label': 'water', 'reason': 'lake'}) as ask:
            gb.repair(args)
        with manifest.open() as stream:
            after = list(csv.DictReader(stream))
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(after[0]['label'], 'built_up')
        self.assertEqual([r['path'] for r in before if r['split'] == 'test'],
                         [r['path'] for r in after if r['split'] == 'test'])
        self.assertTrue(all(r['split'] == 'train' for r in after[len(before):]))

    def test_report_uses_measured_values(self):
        data = {'results': {'nl_only': {'evaluation': {'IN': {'samples': 15, 'accuracy': 0.6, 'macro_f1': 0.5}}}}}
        source = self.root / 'results.json'
        source.write_text(json.dumps(data))
        output = self.root / 'report.md'
        gb.report(argparse.Namespace(results=source, output=output, claude=False))
        self.assertIn('| nl_only | IN | 15 | 0.600 | 0.500 |', output.read_text())


if __name__ == '__main__':
    unittest.main()
