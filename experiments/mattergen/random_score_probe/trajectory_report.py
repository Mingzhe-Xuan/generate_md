"""Show similarity together with force coverage; masked states stay unknown."""
import json
from pathlib import Path


def write_report(root,result):
    root=Path(root)
    rows=result['summary']
    lines=['# Random-prior unconditional MatterGen / MatterSim force probe','',
        f'{result["policy"]["trajectories"]} trajectories; {len(result["states"])} snapshots; '+
        f'{result["raw_model_calls"]} batched score calls; {result["elapsed_seconds"]:.1f}s; MS NFE {result["NFE"]}.',
        '', 'Same dft_band_gap checkpoint, condition disabled, CFG=0; native joint1000-step sampler.',
        'Initial positions/cells/types sampled from official priors. No known crystal templates or final-type backfill.',
        '', '| Stage | step | t | fully known types | valid cosine /16 | cos(score,force) | cos(sampler displacement,force) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        def fmt(value):return 'unknown' if value is None else f'{value:.4f}'
        lines.append(f'| {r["stage"]} | {r["step"]} | {r["t"]:.3f} | {r["all_types_known"]} | '+
            f'{r["cos_score_force"]["n"]} | {fmt(r["cos_score_force"]["mean"])} | {fmt(r["cos_drift_force"]["mean"])} |')
    lines+=['', 'Masked/unsupported types, invalid cells, density or collision guards and failed force points remain unknown, not zero cosine.',
        'Zero atomic force/score directions are stationary/unknown. No force cosine is defined at the fully masked random prior.',
        'Cell and atom types evolve; only atomic coordinate directions are compared here, not lattice stress or discrete type logits.',
        'Early-stage coverage is selected by eligibility; compare coverage before interpreting stage means.',
        'Bootstrap CIs resample trajectories within each stage. Stages are correlated; no pooled independent-snapshot significance claim.',
        'Positive cosine implies a local potential-energy decrease along that direction; it does not prove finite-step/terminal stability.',
        'No MatterSim force is used to steer the actual generation.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    (root/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['stage']=='predictor']
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    steps=[r['step'] for r in selected]
    for name,label in [('cos_score_force','Cartesian score covector'),('cos_drift_force','Cartesian sampler displacement')]:
        eligible=[r for r in selected if r[name]['n']]
        if eligible:
            x=[r['step'] for r in eligible];y=[r[name]['mean'] for r in eligible]
            errors=[[r[name]['mean']-r[name]['CI95'][0] for r in eligible],
                    [r[name]['CI95'][1]-r[name]['mean'] for r in eligible]]
            axes[0].errorbar(x,y,yerr=errors,marker='o',capsize=3,label=label)
    axes[0].axhline(0,color='grey',lw=.7)
    axes[0].set_ylabel('Mean cosine with MatterSim atomic force')
    axes[0].set_ylim(-1.05,1.05)
    axes[0].legend()
    axes[0].set_title('Unconditional MatterGen from random prior: 16 trajectories')
    axes[1].plot(steps,[r['all_types_known'] for r in selected],'-s',label='Fully unmasked types')
    axes[1].plot(steps,[r['cos_score_force']['n'] for r in selected],'-o',label='Valid force comparison')
    axes[1].set_ylabel('Trajectories /16');axes[1].set_ylim(-.5,16.5)
    axes[1].set_xlabel('Reverse diffusion step (0: random prior; 999: near final)')
    axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'trajectory-force.png',dpi=180);fig.savefig(root/'trajectory-force.svg');plt.close(fig)
