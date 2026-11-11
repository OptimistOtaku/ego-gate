import unittest
import numpy as np
import torch
from experiments.stream_gate import Config, Memory, Net, scores, run_one


class StreamTests(unittest.TestCase):
    def test_gradient_score_matches_autograd(self):
        torch.manual_seed(42)
        net = Net(4, 7)
        x, y = torch.rand(3, 4), torch.tensor([0, 3, 9])
        got = scores(net, x, y, 'curiosity')
        expected = []
        for a,b in zip(x,y):
            net.zero_grad()
            torch.nn.functional.cross_entropy(net(a[None]),b[None]).backward()
            g = sum(p.grad.square().sum() for p in net.parameters())
            expected.append(g/(1+g))
        torch.testing.assert_close(got, torch.stack(expected))

    def test_capacity_and_no_model_updates_during_admission(self):
        net = Net(4, 7)
        before = [p.clone() for p in net.parameters()]
        for policy in ('reservoir','balanced','doubt','curiosity','egogate','egogate_audit'):
            m = Memory(5,policy,42)
            for i in range(30):
                m.add(torch.ones(4)*i/30, i%10, False, net)
                self.assertLessEqual(len(m.items),5)
            self.assertEqual(m.seen,30)
        for a,b in zip(before,net.parameters()):
            torch.testing.assert_close(a,b)

    def test_equal_update_and_exposure_budgets(self):
        # Small deterministic fixture tests protocol accounting, not a reported experiment.
        from unittest.mock import patch
        rng = np.random.default_rng(4)
        x = rng.random((100,4),dtype=np.float32)
        y = np.tile(np.arange(10),10)
        cfg = Config(memory=10,batch=8,hidden=8,period=3)
        with patch('experiments.stream_gate.dataset',return_value=(x,y,x,y)):
            rows=[run_one(cfg,'egogate_audit',s,'.',a) for s,a in
                  [('immediate',False),('periodic',False),('immediate',True),('periodic',True)]]
        self.assertEqual(len(set(r['updates'] for r in rows)),1)
        self.assertEqual(len(set(r['training_exposures'] for r in rows)),1)
        self.assertTrue(all(len(r['trace'])==5 for r in rows))
        self.assertTrue(all(r['buffer_examples']==10 for r in rows))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
