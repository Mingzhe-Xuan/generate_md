"""Frozen annealed-bias method and native proof controls."""
PARENTS=tuple(f'mattergen-{n}.vasp' for n in ('0000','0010','0015','0025','0037','0038','0045','0061','0067','0079','0091','0099','0101','0106','0113','0127'))
SEED=20261034
PEAKS=(.1,.3,1.)
FRACTIONS=(.1,.25,.5,1.,.5,.25,.1,0.)
ARMS=('full','reverse','random','zero','fixed')
PROBE_STAGES=(3,4,7)
FIRE=dict(steps=200,fmax=.002,dt=.05,dtmax=.2,maxstep=.05,alpha=.1,finc=1.1,fdec=.5,fa=.99,Nmin=5,backtracks=12,minimum_dt=1e-8)
POLICY=dict(parent_ids=list(PARENTS),seed=SEED,peaks=list(PEAKS),fractions=list(FRACTIONS),arms=list(ARMS),probe_stages=list(PROBE_STAGES),fire=FIRE,
    objective='(MatterSim total energy-reference total energy)/N+lambda*bias',CF_bias='half squared gap error to2.5eV',
    random_bias='conservative fixed linear tilt, initial atomic/cell norm matched to CF gradient',
    proposal_precision='float64_CF;float32_MatterSim',fixed_species=True,reset_velocity_at_each_stage=True,
    atomic_offset_max_A=2.,strain_norm_max=.15,minimum_distance_A=.7,
    final_energy_ceiling_eV_atom=.05,final_volume_bounds=[.65,1.4],final_raw_and_filtered_fmax=.01,CF_formation_upper=-.5,
    fallback='OriginalB+seedB+ALL9zero-bias FIRE/quench endpoints shared by every arm.',
    steering_trajectories=240,fork_quench_attempts=720,seed_quench_attempts=16,total_quench_attempts=736,
    minimum_material_parents=4,material_gain_eV=.1,minimum_mean_advantage_eV=.05,
    directional_controls=['reverse','random','zero'],require_negative_parent_paired_CI95=True,
    stage_gradient_convergence_is_not_physical_stability=True,role='Development16 method; independent confirmation required.')


def weights(arm,peak):
    if arm not in ARMS or peak not in PEAKS:
        raise ValueError('Unknown frozen bias schedule.')
    return [0.]*len(FRACTIONS) if arm=='zero' else [peak]*len(FRACTIONS) if arm=='fixed' else [peak*f for f in FRACTIONS]
