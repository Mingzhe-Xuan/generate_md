"""Counted cached conservative CF bias plus per-atom MatterSim energy."""
import numpy as np
from anneal_math import combine


class Provider:
    def __init__(self,geometry,scorer,client,reference_energy,anchor_q,anchor_loss,slope,cf_callback=None,initial_physics=None,initial_cf=None):
        self.geometry=geometry;self.scorer=scorer;self.client=client
        self.reference_energy=float(reference_energy);self.anchor_q=np.asarray(anchor_q).copy()
        self.anchor_loss=float(anchor_loss);self.slope=np.asarray(slope).copy();self.cf_callback=cf_callback
        self.physics={};self.cf={};self.CF_evaluations=0;self.physics_evaluations=0;self.requests=0
        key=self.anchor_q.tobytes()
        if initial_physics is not None:self.physics[key]=initial_physics
        if initial_cf is not None:self.cf[key]=initial_cf

    def components(self,q,need_cf):
        q=np.asarray(q,dtype=float);key=q.tobytes();self.requests+=1
        if key not in self.physics:
            row=self.client.points(self.geometry,np.asarray([q]))[0]
            gradient=self.geometry.energy_gradient(q,np.asarray(row['forces_eV_A']),np.asarray(row['stress_eV_A3']))
            self.physics[key]=dict(energy_eV=row['energy_eV'],gradient_E_total=self.geometry.project(gradient),
                raw_fmax=float(np.linalg.norm(row['forces_eV_A'],axis=1).max()),record=row)
            self.physics_evaluations+=1
        if need_cf and key not in self.cf:
            if self.cf_callback is not None:
                value=self.cf_callback(self.geometry,q)
            else:
                from anneal_shared import precision_score
                value=precision_score(self.scorer,self.geometry,q,self.scorer.torch.float64,True)
            if not np.isfinite([value['gap'],value['eform'],value['loss']]).all() or not np.isfinite(value['gradient']).all():
                raise ValueError('Nonfinite CF bias component.')
            value=dict(value,gradient=self.geometry.project(value['gradient']))
            self.cf[key]=value;self.CF_evaluations+=1
        return self.physics[key],self.cf.get(key)

    def evaluate(self,q,weight,arm):
        need_cf=weight!=0 and arm in ('full','reverse','fixed')
        p,c=self.components(q,need_cf)
        state=combine(np.asarray(q),arm,weight,p['energy_eV'],p['gradient_E_total'],self.geometry.n,self.reference_energy,
            c,self.anchor_q,self.anchor_loss,self.slope,self.geometry.project)
        state.update(raw_physical_fmax=p['raw_fmax'],CF_gap=c['gap'] if c is not None else None,
            CF_loss=c['loss'] if c is not None else None,physical_gradient_norm_per_atom=float(np.linalg.norm(p['gradient_E_total'])/self.geometry.n),
            weighted_bias_gradient_norm=float(weight*np.linalg.norm(state['bias_gradient'])))
        physical=p['gradient_E_total']/self.geometry.n;steer=weight*state['bias_gradient'];den=np.linalg.norm(physical)*np.linalg.norm(steer)
        state['cosine_physical_vs_bias']=float(physical@steer/den) if den>1e-30 else None
        return state

    def costs(self):
        return dict(physics_point_evaluations=self.physics_evaluations,CF_backward_evaluations=self.CF_evaluations,component_requests=self.requests,
            unique_physical_points=len(self.physics),unique_CF_points=len(self.cf))
