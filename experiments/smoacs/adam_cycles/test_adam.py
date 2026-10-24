import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adam_policy import BASE_RATES,cosine_fraction,flags,freeze_inactive,paired_flags,set_rates,valid_proposal,schedule_position


class AdamCycleTests(unittest.TestCase):
    def test_twenty_update_schedule(self):
        self.assertEqual(cosine_fraction(0,20),1.)
        self.assertAlmostEqual(cosine_fraction(10,20),.5)
        self.assertEqual(cosine_fraction(20,20),0.)
        self.assertEqual(schedule_position(1,19,20,1,'per_phase'),(19,20))
    def test_one_update_per_cycle_uses_global_decay_and_preserves_Adam_through_B(self):
        import torch
        parameters=[torch.nn.Parameter(torch.ones(1)) for _ in range(3)]
        optimizers=[torch.optim.Adam([p],lr=lr) for p,lr in zip(parameters,BASE_RATES)]
        rates=[]
        for cycle in range(1,201):
            position,period=schedule_position(cycle,0,1,200,'global')
            self.assertEqual((position,period),(cycle-1,200))
            set_rates(optimizers,position,period)
            rates.append(optimizers[0].param_groups[0]['lr'])
            for p,o in zip(parameters,optimizers):
                p.grad=torch.ones_like(p)
                o.step()
                momentum=o.state[p]['exp_avg'].clone()
                with torch.no_grad():p.fill_(1.)
                torch.testing.assert_close(momentum,o.state[p]['exp_avg'])
                self.assertEqual(float(o.state[p]['step']),cycle)
            self.assertEqual(schedule_position(cycle,1,1,200,'global'),(cycle,200))
        self.assertEqual(rates[0],BASE_RATES[0])
        self.assertTrue(all(a>b for a,b in zip(rates,rates[1:])))
        self.assertGreater(rates[-1],0)
        self.assertEqual(schedule_position(1,600,600,1,'restart200',phase_end=True),(200,200))
    def test_audit_rejects_CPU_forward_and_never_marks_it_passed(self):
        import torch
        from types import SimpleNamespace
        from gpu import GPUAudit
        with tempfile.TemporaryDirectory() as tmp,patch('gpu.require_cuda'):
            context=dict(gap_model=SimpleNamespace(encoder=torch.nn.Linear(2,1)),ef_model=SimpleNamespace(encoder=torch.nn.Linear(2,1)))
            audit=GPUAudit(context,Path(tmp)/'audit.json')
            with self.assertRaises(RuntimeError):context['gap_model'].encoder(torch.ones(1,2))
            audit.close();self.assertFalse(audit.report['all_cuda']);self.assertFalse(audit.report['passed'])

    def test_submission_dependencies_and_repeat_safety(self):
        import submit
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);destination=root/'results/smoacs/adam-test'
            with patch('pathlib.Path.cwd',return_value=root),patch('sys.argv',['submit','--output',str(destination)]),\
                 patch('submit.subprocess.check_output',side_effect=['100\n','101\n','102\n']) as call:
                submit.main();submit.main()
                self.assertEqual(call.call_count,3)
                self.assertIn('--dependency=afterok:100',call.call_args_list[1].args[0])
                self.assertIn('--dependency=afterok:101',call.call_args_list[2].args[0])
                self.assertEqual(call.call_args_list[0].kwargs['env']['SMOACS_ADAM_RUN'],str(destination))

    def test_device_migration_preserves_forward_closure_aliases(self):
        import torch
        from gpu import move_context,require_cuda
        parameter=torch.nn.Parameter(torch.ones(1,3));base=torch.ones(1,3)
        data=dict(batch=torch.zeros(1,dtype=torch.long),size=torch.ones(1,dtype=torch.long),features=torch.ones(1,3))
        context=dict(targets=[parameter],gap_model=torch.nn.Linear(3,1),ef_model=torch.nn.Linear(3,1),data=data,
            base=base,lower=base.clone(),upper=base.clone(),angle_lower=base.clone(),angle_upper=base.clone())
        optimizer=torch.optim.Adam([parameter]);alias=base
        move_context(context,'cpu',torch.float64)
        self.assertIs(alias,context['base']);self.assertEqual(alias.dtype,torch.float64)
        self.assertIs(parameter,optimizer.param_groups[0]['params'][0])
        self.assertEqual((parameter*alias).dtype,torch.float64)
        self.assertEqual(context['batch'].dtype,torch.long)
        with self.assertRaises(RuntimeError):require_cuda(context)

    def test_schedule_and_600_updates_preserve_moments(self):
        import torch
        self.assertEqual(cosine_fraction(0),1.)
        self.assertAlmostEqual(cosine_fraction(100),.5)
        self.assertGreater(cosine_fraction(199),0.)
        self.assertEqual(cosine_fraction(200),0.)
        parameters=[torch.nn.Parameter(torch.ones(2,1,dtype=torch.float64)) for _ in range(3)]
        optimizers=[torch.optim.Adam([p],lr=lr) for p,lr in zip(parameters,BASE_RATES)]
        mask=torch.tensor([[True],[False]])
        for cycle in range(3):
            for step in range(200):
                set_rates(optimizers,step)
                if step==0:self.assertEqual([o.param_groups[0]['lr'] for o in optimizers],list(BASE_RATES))
                for p,o in zip(parameters,optimizers):
                    before=[p.detach().clone()];p.grad=torch.ones_like(p);o.step();freeze_inactive(o,[p],before,[mask])
                    self.assertEqual(float(p[1].detach()),1.)
            moment=optimizers[0].state[parameters[0]]['exp_avg'].clone()
            with torch.no_grad():parameters[0][0].add_(.1)
            torch.testing.assert_close(moment,optimizers[0].state[parameters[0]]['exp_avg'])
        self.assertEqual(float(optimizers[0].state[parameters[0]]['step']),600.)
        state=optimizers[0].state[parameters[0]];state['exp_avg'][1]=float('nan')
        freeze_inactive(optimizers[0],[parameters[0]],[parameters[0].detach().clone()],[mask])
        self.assertTrue(torch.isfinite(state['exp_avg']).all())

    def test_thresholds_and_unknown_paired_denominators(self):
        self.assertTrue(flags(2.3,-.6,True)['joint']);self.assertTrue(flags(2.7,-.6,True)['joint'])
        self.assertFalse(flags(2.5,-.5,True)['joint']);self.assertFalse(flags(2.5,-.6,False)['joint'])
        ids=['a','b','c','d'];first=dict(zip(ids,[True,False,None,False]));third=dict(zip(ids,[False,True,True,None]))
        report=paired_flags(first,third,ids)
        self.assertEqual(report['pass_to_fail'],1);self.assertEqual(report['fail_to_pass'],1);self.assertEqual(report['unknown'],2)
        self.assertEqual(report['cycle1']['denominator'],4);self.assertEqual(report['possible_difference_pp'],[0.,50.])
        with self.assertRaises(ValueError):paired_flags(first,{},ids)

    def test_singular_or_nonfinite_proposal_is_not_exported(self):
        import torch
        self.assertTrue(valid_proposal(torch.tensor(1.),torch.eye(3),torch.zeros(2,3)))
        self.assertFalse(valid_proposal(torch.tensor(1.),torch.zeros(3,3),torch.zeros(2,3)))
        self.assertFalse(valid_proposal(torch.tensor(float('nan')),torch.eye(3),torch.zeros(2,3)))

    def test_export_does_not_carry_cycle1_B_success_into_missing_cycle3(self):
        import export
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ids=['a.vasp','b.vasp','c.vasp','d.vasp'];cycles=[]
            for cycle in (1,2,3):
                rows=[]
                for identifier in ids:
                    path=root/(str(cycle)+identifier);path.write_text('A geometry '+str(cycle))
                    a=dict(candidate_id=identifier,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        num_atoms=2,formula='NaCl',elements=['Na','Cl'],gap=2.5,eform=-1.,neutral=True,separated=True,valid=True)
                    rows.append(dict(candidate_id=identifier,cycle=cycle,A=a,a_updates=200,status='not_eligible'))
                cycles.append(dict(cycle=cycle,candidates=rows))
            b=root/'B.vasp';b.write_text('B geometry')
            cycles[0]['candidates'][0].update(status='completed',b_result=dict(candidate_id=ids[0],status='converged',
                input_sha256=cycles[0]['candidates'][0]['A']['sha256'],relaxed_path=str(b),relaxed_sha256=hashlib.sha256(b.read_bytes()).hexdigest()))
            (root/'complete.json').write_text(json.dumps(dict(completed=True)))
            (root/'results.json').write_text(json.dumps(dict(protocol=dict(ids=ids),cycles=cycles)))
            with patch.dict(os.environ,SLURM_JOB_ID='unit'),patch('sys.argv',['export','--run',str(root)]):export.main()
            inventory=json.loads((root/'evaluation-cycle3/inventory-249/inventory.json').read_text())
            self.assertEqual(len(inventory['candidates']),4)
            self.assertEqual(json.loads((root/'evaluation-cycle3/relaxation/a/result.json').read_text())['status'],'not_eligible')
            scope=json.loads((root/'reference-scope.json').read_text());self.assertEqual(len(scope['candidates']),1)

    def test_report_unknowns_and_same_ID_checks(self):
        from report import paired_report
        ids=['a','b','c','d'];one={i:dict(structure_valid=True,crystalformer_flags=flags(2.5,-1.,True),alignn_flags=flags(2.5,-1.,True)) for i in ids}
        three=copy.deepcopy(one);three['a']['alignn_flags']=flags(None,None,False)
        report=paired_report(one,three,ids)
        self.assertEqual(report['alignn_joint']['cycle3']['unknown_count'],1)
        self.assertEqual(report['crystalformer_joint']['cycle3']['pass_count'],4)
        with self.assertRaises(ValueError):paired_report(one,{},ids)


if __name__=='__main__':unittest.main()
