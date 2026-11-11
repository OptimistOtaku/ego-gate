import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.analyze_offline_benchmark import analyze, BUDGETS, METHODS


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (Path(__file__).resolve().parents[1]/'tmp').mkdir(exist_ok=True)

    def test_incomplete_grid_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as td:
            root=Path(td)
            (root/'manifest.json').write_text(json.dumps(dict(gpu='test',device='cuda:0',smoke=False,source_sha256='test')))
            (root/'COMPLETE.json').write_text('{}')
            with self.assertRaisesRegex(AssertionError,'Incomplete bounded grid'):
                analyze(root,root/'out')

    def test_full_synthetic_grid_and_paired_direction(self):
        # Fixture is deliberately synthetic and never written to evidence directories.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as td:
            root=Path(td)
            (root/'manifest.json').write_text(json.dumps(dict(gpu='test',device='cuda:0',smoke=False,source_sha256='test')))
            (root/'COMPLETE.json').write_text('{}')
            for d,bs in BUDGETS.items():
                for seed in range(100,110):
                    rows=[]
                    for b,m in [(b,m) for b in bs for m in METHODS]+[(0,'none'),(600,'full')]:
                        ix=list(range(b))
                        acc=.5 + ((seed-99)*.001 if m=='egogate' else 0.)
                        row=dict(dataset=d,seed=seed,budget=b,method=m,task_a_pre=.8,
                                 task_a_post=acc,task_b_post=acc,final_accuracy=acc,forgetting=.8-acc,
                                 selection_seconds=.1,training_seconds=.2,base_seconds=.1,
                                 memory_bytes=b*12,cuda_peak_bytes=10000,coverage=min(b,5),
                                 selected_indices=ix,class_counts=[b]+[0]*9,
                                 selection_sha256=hashlib.sha256(np.asarray(ix,dtype='<i8').tobytes()).hexdigest(),
                                 updates=10,exposures=100,replay_exposures=20)
                        rows.append(row)
                    (root/f'{d}_{seed}.json').write_text(json.dumps(dict(complete=True,dataset=d,rows=rows)))
            analyze(root,root/'out')
            audit=json.loads((root/'out/audit.json').read_text())
            self.assertEqual(audit['rows'],870)
            self.assertEqual(audit['primary_tests'],27)
            import csv
            with (root/'out/primary_comparisons.csv').open() as handle:
                comparisons=list(csv.DictReader(handle))
            self.assertTrue(all(abs(float(r['accuracy_delta_pp'])-.55)<1e-10 for r in comparisons))
            self.assertTrue(all(float(r['holm_p_value'])>=float(r['p_value']) for r in comparisons))


if __name__=='__main__':
    unittest.main()
