"""Frozen development16 post-quench comparison, separate from native proofs."""
PARENTS = tuple(f'mattergen-{n}.vasp' for n in ('0000','0010','0015','0025','0037','0038','0045','0061','0067','0079','0091','0099','0101','0106','0113','0127'))
TIMES = (.1, .3)
SCALES = (.05, .2, .5, .8)
ARMS = ('full', 'reverse', 'random', 'stop', 'direct')
OUTPUTS = ARMS + ('seeds', 'baseline')
POLICY = dict(parent_ids=list(PARENTS), seed=20261032, times=list(TIMES), scales=list(SCALES), arms=list(ARMS),
    proposal_precision='float64', final_CF_precision='official_float32', fixed_species=True,CF_loss='half_squared_gap_error; formation is a screening constraint',
    geometry_arithmetic='difference-before-offset', guidance_scale=2.,
    fallback='Every arm shares originalB and BOTH denoised-seed-B candidates; each arm adds its own8 rays.',
    attempts=672, seed_attempts=32, ray_attempts=640,
    force_fmax=.01, raw_fmax=.01, steps=1600, maxstep=.05, preserve_symmetry=True,
    target_gap=2.5, CF_formation_upper=-.5, original_energy_ceiling_eV_atom=.05,
    original_volume_bounds=[.65,1.4], proposal_min_distance_A=.7,
    proposal_max_atomic_A=2., proposal_max_strain=.15,
    material_gain_eV=.1, minimum_material_parents=4, minimum_mean_advantage_eV=.05,
    directional_controls=['reverse','random'], Jacobian_controls=['stop','direct'],
    require_negative_parent_paired_CI95=True,
    role='Development16, not unseen-parent confirmation. Denoised seeds alone cannot establish gradient value.')
