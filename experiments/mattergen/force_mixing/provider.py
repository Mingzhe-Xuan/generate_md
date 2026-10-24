"""Combine current-pose MatterSim energy gradients with CF property gradients."""
import json
import torch

from geometry import combine,energy_surrogate,geometry,property_cartesian_force


class MixedGradient:
    def __init__(self,context,client,weight):
        self.context=context
        self.client=client
        self.weight=weight
        self.calls=0
        self.output=context['output']
        (self.output/'force-snapshots').mkdir()

    def tensors(self,cells,frac,active):
        c=self.context
        numbers=c['data']['fixed_atomic_distribution'].argmax(1)+1
        rows=self.client.evaluate(c['ids'],cells,frac,numbers,c['slices'],active)
        energy=torch.zeros(len(c['ids']),dtype=cells.dtype,device=cells.device)
        force=torch.zeros_like(frac)
        stress=torch.zeros_like(cells)
        for i,key in enumerate(c['ids']):
            if key not in rows:
                continue
            row=rows[key]
            energy[i]=row['energy_eV']
            force[c['slices'][i]]=torch.tensor(row['forces_eV_A'],dtype=cells.dtype,device=cells.device)
            stress[i]=torch.tensor(row['stress_eV_A3'],dtype=cells.dtype,device=cells.device)
        if not all(torch.isfinite(v).all() for v in (energy,force,stress)):
            raise ValueError('Invalid physical tensors.')
        return energy,force,stress,rows

    def __call__(self,property_gradients,loss,active,cycle,step):
        c=self.context
        cells,frac=geometry(c)
        energy,force,stress,rows=self.tensors(cells,frac,active)
        denominator=c.get('gradient_denominator',len(c['ids']))
        surrogate=energy_surrogate(cells,frac,c['batch'],energy,force,stress)
        value=surrogate[active].sum()/denominator+c['targets'][3].sum()*0+c['targets'][4].sum()*0
        physical=torch.autograd.grad(value,c['targets'])
        mixed=combine(physical,property_gradients,self.weight)
        if not all(g.is_cuda and torch.isfinite(g).all() for g in mixed):
            raise ValueError('Mixed gradients must be finite CUDA values.')
        self.calls+=1
        f1=property_cartesian_force(property_gradients[2],cells,c['batch'],denominator)
        cases=[]
        for i,key in enumerate(c['ids']):
            if key not in rows:
                continue
            a,b=f1[c['slices'][i]],force[c['slices'][i]]
            norm1,norm2=float(a.norm()),float(b.norm())
            cosine=float((a*b).sum())/(norm1*norm2) if norm1>1e-12 and norm2>1e-12 else None
            if cosine is not None:
                cosine=max(-1.,min(1.,cosine))
            ratio=self.weight*norm1/norm2 if norm2>1e-12 else None
            cases.append(dict(candidate_id=key,pose_sha256=rows[key]['pose_sha256'],energy_total_eV=float(energy[i]),
                CF_loss=float(loss[i]),combined_loss=float(energy[i])+self.weight*float(loss[i]),
                F1_RMS_eV_A=norm1/len(a)**.5,F2_RMS_eV_A=norm2/len(b)**.5,
                coordinate_force_cosine=cosine,lambda_F1_over_F2=ratio,
                max_atomic_force=float(torch.linalg.vector_norm(b,dim=1).max()),max_abs_stress=float(stress[i].abs().max())))
        record=dict(cycle=cycle,A_step=step+1,weight=self.weight,denominator=denominator,cases=cases,
            max_abs_gradient_blocks=[float(g.abs().max()) for g in mixed],
            max_abs_physical_blocks=[float(g.abs().max()) for g in physical],
            max_abs_property_blocks=[float(g.abs().max()) for g in property_gradients])
        with (self.output/'force-trace.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        if step==0 or (step+1)%25==0:
            torch.save(dict(cycle=cycle,A_step=step+1,weight=self.weight,denominator=denominator,
                cells=cells.detach().cpu(),frac=frac.detach().cpu(),batch=c['batch'].cpu(),active=active.cpu(),
                energies=energy.detach().cpu(),forces=force.detach().cpu(),stress=stress.detach().cpu(),
                property_gradients=[g.detach().cpu() for g in property_gradients[:3]],
                physical_gradients=[g.detach().cpu() for g in physical[:3]],
                mixed_gradients=[g.detach().cpu() for g in mixed[:3]],
                targets=[p.detach().cpu() for p in c['targets'][:3]],
                base=c['base'].cpu(),lower=c['lower'].cpu(),upper=c['upper'].cpu(),
                angle_lower=c['angle_lower'].cpu(),angle_upper=c['angle_upper'].cpu(),
                pose_hashes={k:r['pose_sha256'] for k,r in rows.items()}),
                self.output/'force-snapshots'/f'step-{step+1:03d}.pt')
        return mixed

    def finish(self,cycle,points):
        c=self.context
        with torch.no_grad():
            cells,frac=geometry(c)
            energy,force,stress,rows=self.tensors(cells,frac,torch.ones(len(c['ids']),dtype=torch.bool,device=cells.device))
        final=dict(cycle=cycle,gradient_calls=self.calls,weight=self.weight,worker_metadata=self.client.ready,
            endpoint=[dict(candidate_id=key,pose_sha256=rows[key]['pose_sha256'],energy_total_eV=float(energy[i]),
                max_raw_force=float(torch.linalg.vector_norm(force[c['slices'][i]],dim=1).max()),
                max_abs_stress=float(stress[i].abs().max()),A_sha256=points[i]['sha256']) for i,key in enumerate(c['ids'])])
        (self.output/'force-mixing-audit.json').write_text(json.dumps(final,indent=2)+'\n')
        self.client.close()
