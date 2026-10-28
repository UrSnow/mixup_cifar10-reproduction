import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from mixup_repro.core import improves, learning_rate, mixed_loss, stratified_split, weights_hash
from mixup_repro.experiment import (ResNet18, evaluate_model, freeze_runs,
                                   seed_everything, train_run, test_frozen)
from vendor.resnet import PreActBlock


class TinyData(Dataset):
    def __len__(self): return 8
    def __getitem__(self,index):
        # Deliberately consumes global RNG to check independence from mixup RNG.
        return torch.ones(3,4,4)*index/8 + torch.rand(3,4,4)*.01, index%2


def tiny_model():
    return nn.Sequential(nn.Flatten(),nn.Linear(48,2))


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_stratification_and_no_leakage(self):
        labels=np.repeat(np.arange(10),5000)
        split=stratified_split(labels,500,2026)
        self.assertEqual(split,stratified_split(labels,500,2026))
        self.assertEqual(len(split['train']),45000)
        self.assertEqual(len(split['validation']),5000)
        self.assertFalse(set(split['train']) & set(split['validation']))
        self.assertEqual(set(split['train']) | set(split['validation']),set(range(50000)))
        np.testing.assert_array_equal(np.bincount(labels[split['validation']]),[500]*10)

    def test_mixup_loss_and_gradient_match_soft_targets(self):
        x=torch.randn(4,3,requires_grad=True)
        y=torch.tensor([0,1,2,0]); other=y.flip(0); lam=.7
        soft=lam*nn.functional.one_hot(y,3)+(1-lam)*nn.functional.one_hot(other,3)
        actual=mixed_loss(x,y,other,lam)
        expected=-(soft*nn.functional.log_softmax(x,1)).sum(1).mean()
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0])

    def test_epoch_boundaries_and_tie_rule(self):
        cfg={'lr':.1,'milestones':[25,38]}
        for epoch,expected in [(1,.1),(25,.1),(26,.01),(38,.01),(39,.001),(50,.001)]:
            self.assertAlmostEqual(learning_rate(cfg,epoch),expected)
        self.assertFalse(improves(5.,5.))
        self.assertTrue(improves(4.9,5.))

    def test_official_architecture_and_initialization(self):
        seed_everything(1); first=ResNet18()
        self.assertIsInstance(first.layer1[0],PreActBlock)
        self.assertEqual(sum(isinstance(m,PreActBlock) for m in first.modules()),8)
        seed_everything(1); second=ResNet18()
        self.assertEqual(weights_hash(first.state_dict()),weights_hash(second.state_dict()))
        first.eval()
        with torch.inference_mode(): self.assertEqual(tuple(first(torch.randn(2,3,32,32)).shape),(2,10))
        first.train()
        loss=mixed_loss(first(torch.randn(2,3,32,32)),torch.tensor([0,1]),torch.tensor([1,0]),.4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in first.parameters() if p.grad is not None))

    @patch('mixup_repro.experiment.environment',return_value={'test':'synthetic'})
    def test_resume_matches_uninterrupted_and_pairs_start_identically(self,_):
        cfg={'epochs':3,'batch_size':4,'lr':.1,'momentum':.9,'weight_decay':.0001,
             'milestones':[1,2],'alpha':1.,'seeds':[1],'num_workers':0}
        device=torch.device('cpu')
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            full=train_run(cfg,'mixup',1,TinyData(),TinyData(),'synthetic',base/'full',device,model_factory=tiny_model)
            train_run(cfg,'mixup',1,TinyData(),TinyData(),'synthetic',base/'resume',device,stop_after=1,model_factory=tiny_model)
            resumed=train_run(cfg,'mixup',1,TinyData(),TinyData(),'synthetic',base/'resume',device,resume=True,model_factory=tiny_model)
            a=torch.load(full,weights_only=False); b=torch.load(resumed,weights_only=False)
            self.assertEqual(weights_hash(a['model']),weights_hash(b['model']))
            self.assertEqual(a['best_epoch'],b['best_epoch'])
            self.assertEqual(weights_hash(a['best_model']),weights_hash(b['best_model']))
            erm=train_run(cfg,'erm',1,TinyData(),TinyData(),'synthetic',base/'erm',device,model_factory=tiny_model)
            self.assertEqual(a['initial_weights_sha256'],torch.load(erm,weights_only=False)['initial_weights_sha256'])
            changed=copy.deepcopy(cfg); changed['alpha']=.5
            with self.assertRaisesRegex(RuntimeError,'Resume rejected'):
                train_run(changed,'mixup',1,TinyData(),TinyData(),'synthetic',base/'resume',device,resume=True,model_factory=tiny_model)

    @patch('mixup_repro.experiment.environment',return_value={'test':'synthetic'})
    def test_freeze_rejects_incomplete_runs_and_blocks_later_training(self,_):
        from mixup_repro.core import digest
        cfg={'epochs':2,'batch_size':4,'lr':.1,'momentum':.9,'weight_decay':.0001,
             'milestones':[1],'alpha':1.,'seeds':[1],'num_workers':0}
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp); split={'synthetic':True}; (out/'split.json').write_text(json.dumps(split))
            for m in ('erm','mixup'):
                train_run(cfg,m,1,TinyData(),TinyData(),digest(split),out,torch.device('cpu'),stop_after=1,model_factory=tiny_model)
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                freeze_runs(cfg,out)
            for m in ('erm','mixup'):
                train_run(cfg,m,1,TinyData(),TinyData(),digest(split),out,torch.device('cpu'),resume=True,model_factory=tiny_model)
            freeze_runs(cfg,out)
            with self.assertRaisesRegex(RuntimeError,'frozen'):
                train_run(cfg,'erm',1,TinyData(),TinyData(),digest(split),out,torch.device('cpu'),resume=True,model_factory=tiny_model)

    @patch('mixup_repro.experiment.environment',return_value={'test':'synthetic'})
    def test_evaluation_gate_and_validation_selected_weights(self,_):
        from mixup_repro.core import digest
        cfg={'epochs':1,'batch_size':4,'lr':.1,'momentum':.9,'weight_decay':.0001,
             'milestones':[],'alpha':1.,'seeds':[1],'num_workers':0,
             'mean':[.5]*3,'std':[.5]*3}
        device=torch.device('cpu')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            with patch('mixup_repro.experiment.datasets.CIFAR10') as cifar:
                with self.assertRaises(FileNotFoundError): test_frozen(out,'unused',device)
                cifar.assert_not_called()
            split={'synthetic':True}; (out/'split.json').write_text(json.dumps(split))
            expected=[]
            for method in ('erm','mixup'):
                path=train_run(cfg,method,1,TinyData(),TinyData(),digest(split),out,device,model_factory=tiny_model)
                state=torch.load(path,weights_only=False)
                expected.append(weights_hash(state['best_model']))
                # Make last weights different: evaluator must still load selected best weights.
                state['model']={k:torch.zeros_like(v) for k,v in state['model'].items()}
                torch.save(state,path)
            freeze_runs(cfg,out)
            observed=[]
            def metric(model,*args):
                observed.append(weights_hash(model.state_dict()))
                return {'loss':1.,'error_pct':50.,'correct':5000,'count':10000}
            with patch('mixup_repro.experiment.datasets.CIFAR10') as cifar, \
                 patch('mixup_repro.experiment.ResNet18',side_effect=tiny_model), \
                 patch('mixup_repro.experiment.evaluate_model',side_effect=metric):
                cifar.return_value.__len__.return_value=10000
                test_frozen(out,'unused',device)
                self.assertFalse(cifar.call_args.kwargs['train'])
                self.assertFalse(cifar.call_args.kwargs['download'])
                self.assertEqual(observed,expected)
                cifar.reset_mock()
                with self.assertRaises(FileExistsError): test_frozen(out,'unused',device,download=True)
                cifar.assert_not_called()
                with (out/'erm_seed1/checkpoint.pt').open('ab') as stream: stream.write(b'changed')
                with self.assertRaisesRegex(RuntimeError,'Checkpoint changed'): test_frozen(out,'unused',device)
                cifar.assert_not_called()

    def test_summary_uses_paired_measured_errors(self):
        from mixup_repro.core import atomic_json, file_hash
        from mixup_repro.summarize import summarize
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            atomic_json(out/'frozen.json',{'config':{'seeds':[1,2,3]}})
            rows=[]
            for seed,erm,mix in [(1,10.,8.),(2,12.,9.),(3,11.,12.)]:
                for method,error in [('erm',erm),('mixup',mix)]:
                    rows.append({'seed':seed,'method':method,'error_pct':error})
                    directory=out/f'{method}_seed{seed}'; directory.mkdir()
                    (directory/f'{method}_seed{seed}_history.csv').write_text('epoch,train_objective,validation_error_pct\n1,1.5,50\n2,1.0,40\n')
            atomic_json(out/'test_results.json',{'manifest_sha256':file_hash(out/'frozen.json'),'runs':rows})
            summarize(out)
            report=(out/'summary.md').read_text()
            self.assertIn('| erm | 11.000 | 1.000 |',report)
            self.assertIn('| mixup | 9.667 | 2.082 |',report)
            self.assertIn('Mean paired improvement: 1.333',report)
            self.assertIn('Mixup wins: 2/3',report)
            self.assertTrue((out/'learning_curves.png').stat().st_size>1000)


if __name__=='__main__': unittest.main()
