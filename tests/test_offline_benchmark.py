import unittest
import numpy as np
import torch
from torch.nn import functional as F
from experiments.offline_benchmark import CNN, MLP, METHODS, herding, score_pool, select, train


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_gradient_for_both_architectures(self):
        torch.manual_seed(7)
        for model, x in [(MLP(4, 5), torch.rand(3,4)), (CNN(), torch.rand(2,3,32,32))]:
            y = torch.arange(len(x))
            actual = score_pool(model, x, y, 'curiosity')['curiosity']
            expected = []
            for bx, by in zip(x, y):
                grads = torch.autograd.grad(F.cross_entropy(model(bx[None]), by[None]), tuple(model.parameters()))
                expected.append(sum(g.square().sum().item() for g in grads))
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)

    def test_all_selectors_and_herding_constraints(self):
        torch.manual_seed(9)
        model, x, y = MLP(4,5), torch.rand(60,4), torch.arange(60)%5
        for method in METHODS:
            selected = select(model,x,y,method,13,9)
            self.assertEqual(len(selected),13,method)
            self.assertEqual(len(np.unique(selected)),13,method)
            self.assertTrue(np.all((selected>=0)&(selected<len(y))),method)
            if method in ('herding','stratified_random'):
                counts=np.bincount(y[selected].numpy())
                self.assertLessEqual(counts.max()-counts.min(),1)

    def test_herding_minimizes_stated_greedy_objective(self):
        z=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[.8,.6]])
        selected=herding(z,np.zeros(5),3)
        running=np.zeros(2)
        for i,j in enumerate(selected):
            distance=((z.mean(0)-(z+running)/(i+1))**2).sum(1)
            distance[selected[:i]]=np.inf
            self.assertEqual(j,distance.argmin())
            running+=z[j]

    def test_equal_exposures_across_memory_sizes(self):
        cfg=dict(lr=.1,momentum=0.,batch=8,replay=2)
        x,y=torch.rand(25,4),torch.arange(25)%5
        outputs=[]
        for k in (3,10):
            outputs.append(train(MLP(4,5),x,y,cfg,9,2,x[:k],y[:k]))
        self.assertEqual(outputs[0],outputs[1])
        self.assertEqual(outputs[0],dict(updates=8,exposures=66,replay_exposures=16))


if __name__=='__main__':
    unittest.main()
