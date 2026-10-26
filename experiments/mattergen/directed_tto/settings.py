"""Frozen stage-one direction test, independent of post-relaxation outcomes."""
import random

FAMILIES=('plain','soft','unrolled8')
ARMS=('descent','reverse','random')
SCALES=(.05,.20,.50,.80)
SEED=20261026
POLICY=dict(target_gap=2.5,CF_eform_upper=-.5,final_fmax=.01,raw_fmax=.01,
    common_seed_atomic_RMS_A=.03,common_seed_strain=.003,
    energy_ceiling_eV_atom=.05,volume_ratio_bounds=[.65,1.4],minimum_distance_A=.5,
    proposal_distance_A=.7,proposal_max_atomic_offset_A=2.,proposal_max_strain=.15,
    minimum_retained_gain_eV=.1,min_improved_parents=4,
    min_gradient_advantage_eV=.05,require_paired_CI_better_than_random_and_reverse=True,
    independent_confirmation_required=True,CF_FD_absolute_tolerance=.001,CF_FD_relative_tolerance=.02,
    CF_FD_steps_A=[1e-5,1e-6],CF_FD_arithmetic='float64_same_frozen_weight_values',
    CF_precision_forward_absolute_max=1e-5,CF_precision_gradient_relative_max=.001,
    physical_and_unrolled_FD_steps_A=[.004,.002],CF_coarse_FD_diagnostics_A=[.004,.002],
    energy_FD_absolute_tolerance=.005,energy_FD_relative_tolerance=.01,
    unrolled_FD_absolute_tolerance=.002,unrolled_FD_relative_tolerance=.05,
    Hessian_epsilon_A=.002,Hessian_asymmetry_relative_max=.02,
    lookahead_steps=8,lookahead_eta_spectral_fraction=.3,
    soft_ridge_min_eV_A2=.1,soft_ridge_spectral_fraction=.01)


def choose(ids,neutral,converged,seed=SEED):
    if len(ids)!=128 or len(set(ids))!=128:raise ValueError('Expected complete128 inventory.')
    allowed=[i for i in ids if i in neutral and i in converged]
    if len(allowed)<16:raise ValueError('Insufficient strict-neutral direct-B comparators.')
    selected=set(random.Random(seed).sample(allowed,16))
    return [i for i in ids if i in selected]


def interval(values):
    import numpy as np
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(42)
    return np.quantile(values[rng.integers(0,len(values),(10000,len(values)))].mean(1),[.025,.975]).tolist()


def derivative_checks_pass(records):
    expected={(kind,block,eps) for kind,steps,blocks in (
        ('CF_gradient',POLICY['CF_FD_steps_A'],('atomic','strain','both')),
        ('physical_gradient',POLICY['physical_and_unrolled_FD_steps_A'],('atomic','strain','both')),
        ('unrolled8_gradient',POLICY['physical_and_unrolled_FD_steps_A'],('both',))) for block in blocks for eps in steps}
    primary=[r for r in records if r['kind']!='CF_gradient_coarse_diagnostic']
    actual={(r['kind'],r['block'],r['epsilon']) for r in primary}
    return bool(actual==expected and len(primary)==len(expected) and all(r['passed'] for r in primary))


def assess_gradient(g,random_control,reverse,baseline,seed_control,eligible):
    import numpy as np
    vectors=[np.asarray(v) for v in (g,random_control,reverse,baseline,seed_control,eligible)]
    if any(v.shape!=(16,) for v in vectors):raise ValueError('All16 paired parents required.')
    g,r,v,b,s,valid=vectors
    comparisons={name:dict(mean_delta=float((g-other).mean()),CI95=interval(g-other)) for name,other in (('random',r),('reverse',v))}
    count=int(sum((np.minimum(b,s)-g>=.1)&valid.astype(bool)))
    success=count>=4 and all(c['mean_delta']<=-.05 and c['CI95'][1]<0 for c in comparisons.values())
    return dict(retained_gains_ge_0p1=count,MAE_vs_baseline_delta=float((g-b).mean()),MAE_vs_baseline_CI95=interval(g-b),
        gradient_vs_controls=comparisons,discovery_gradient_advantage=bool(success),confirmation_needed=bool(success))
