"""Pre-outcome development protocol for CF-guided post-quench gradient estimation."""
PARENTS=tuple(f'mattergen-{i:04d}.vasp' for i in (0,10,15,25,37,38,45,61,67,79,91,99,101,106,113,127))
FAMILIES=('cf_guided','random_guided','isotropic')
ARMS=('descent','reverse','random','raw_cf')
SCALES=(.2,.5,.8,1.2)
SEED=20261031
POLICY=dict(parent_count=16,search_seed=SEED,families=list(FAMILIES),arms=list(ARMS),scales=list(SCALES),
    pairs=4,sigma_RMS_A=.8,guided_alpha=.5,isotropic_alpha=1.,
    metric='Translation-free atomic RMS offsets plus orthonormal symmetric cell strain, unchanged Geometry.',
    random_guide='Matches the CF guide atomic/strain block norms; all families share Gaussian/scalar draws.',
    loss='Full canonical B, then CFgapMAE/(1+CFgapMAE); physical/formation/geometry failures have utility1.',
    selection='All four arms in each family share originalB, seedB and all eight probe endpoints as fallback; Best4 heldout rays added per arm.',
    invalid='Every geometric rejection, stationary direction and failed prediction/quench consumes its predeclared attempt; no resampling.',
    calls=dict(parents=16,shared_seed_attempts=16,probe_attempts=384,heldout_ray_attempts=768,total_attempts=1152),
    physics=dict(fmax=.01,raw_fmax=.01,maxstep=.05,cap=1600,preserve_symmetry=True,CF_eform_upper=-.5,
        original_energy_ceiling_eV_atom=.05,original_volume_ratio=[.65,1.4]),
    discovery=dict(min_material_parents=4,minimum_gain_beyond_probes_eV=.1,min_mean_advantage_eV=.05,
        require_negative_parent_CI_both_direction_controls=True),
    prior_importance='Separately require final AND beyond-probe incremental advantages >=.05 with negative pairedCI versus isotropic and random-guided ES.',
    heldout='ALIGNN never ranks proposals. Original16 are development, not independent confirmation. Physical retention and new confirmation remain necessary.',
    reference='https://proceedings.mlr.press/v97/maheswaranathan19a.html')
OUTPUTS=tuple(f'{family}-{arm}' for family in FAMILIES for arm in ARMS)+tuple(f'{family}-probes' for family in FAMILIES)+('baseline',)
