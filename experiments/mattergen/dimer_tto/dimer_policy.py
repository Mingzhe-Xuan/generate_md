"""Prespecified small real-physics feasibility study; not population efficacy."""
import numpy as np

PARENTS=('mattergen-0000.vasp','mattergen-0010.vasp','mattergen-0015.vasp','mattergen-0025.vasp')
ARMS=('descent','reverse','random')
POLICY=dict(parent_ids=list(PARENTS),seed=20261036,fixed_species=True,activation_cell='fixed seed cell',
    CF='Previously validated GPU F64 gap-loss gradient at the identical441 seed; initial direction only.',
    initial_RMS_A=.10,exit_RMS_A=.05,steps=120,max_num_rot=5,max_translation_A=.05,
    dimer_separation_RMS_A=.002,rotation_fmin=.001,rotation_fmax=.01,saddle_raw_fmax=.02,
    negative_curvature_threshold_eV_A2=-.01,Hessian_RMS_steps_A=[.004,.002],Hessian_asymmetry_max=.02,
    Hessian_agreement_absolute=.05,Hessian_agreement_relative=.05,source_curvature_absolute=.02,source_curvature_relative=.05,
    max_offset_A=2.,minimum_distance_A=.7,intermediate_energy_ceiling_eV_atom=1.,NFE_cap_per_activation=2000,
    saddle_attempts=12,future_quench_slots=24,failures_consume_slots=True,
    role='First4 physical saddle-search feasibility only. No final crystal property, population or goal completion proof.')


def directions(gradient,natoms,parent_index):
    g=np.asarray(gradient,dtype=float)
    if natoms<2 or g.shape!=(3*natoms+6,) or not np.isfinite(g).all():raise ValueError('Invalid cached CF gradient.')
    d=-g[:-6].reshape(natoms,3)/np.sqrt(natoms);d-=d.mean(0);norm=np.linalg.norm(d)
    if norm<1e-10:return None
    d/=norm;r=np.random.default_rng([POLICY['seed'],parent_index]).normal(size=d.shape);r-=r.mean(0);r/=np.linalg.norm(r)
    return dict(descent=d,reverse=-d,random=r)


def ledger(parents):
    return [dict(id=f'{p.removesuffix(".vasp")}-{a}-exit{sign:+d}',parent_id=p,arm=a,exit_sign=sign)
        for p in parents for a in ARMS for sign in (-1,1)]
