"""Plot cumulative paired effort, not only the final tighter continuation."""
import argparse
import json
from pathlib import Path


def draw(root,report,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows={(r['trajectory'],r['arm']):r for r in data['relaxations']}
    def steps(i,arm):
        row=rows[i,arm]
        return row['result']['steps']+row.get('coarse_result',{}).get('steps',0)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for arm,color in (('mean800','#1769a4'),('sample800','#d06923')):
        cases=[c for c in report['pairs'][arm]['cases'] if c['status']=='compared']
        axes[0].scatter([c['trajectory'] for c in cases],[1000*c['delta_energy_eV_atom'] for c in cases],label=arm,color=color,alpha=.75)
        axes[1].scatter([steps(c['trajectory'],'mean1000') for c in cases],[steps(c['trajectory'],arm) for c in cases],label=arm,color=color,alpha=.75)
    axes[0].axhline(0,color='k',ls='--',lw=.7)
    axes[0].set(title='Final paired MatterSim energy',xlabel='Parent trajectory ID',ylabel='E(800 + relax) - E(1000 + relax), meV/atom')
    maximum=max(axes[1].get_xlim()[1],axes[1].get_ylim()[1]);axes[1].plot([0,maximum],[0,maximum],'k--',lw=.7)
    axes[1].set(title='Cumulative coarse + tight BFGS effort',xlabel='1000-step branch: total BFGS steps',ylabel='800-step branch: total BFGS steps')
    arms=['mean800','sample800','mean1000'];branches=report['branches']
    generation=[branches[a]['native_generation_seconds'] for a in arms]
    relax=[branches[a].get('cumulative_relaxation_seconds',branches[a]['attempted_seconds']) for a in arms]
    axes[2].bar(arms,generation,label='Native sampling prefix',color='#80a8c5')
    axes[2].bar(arms,relax,bottom=generation,label='All relaxation work',color='#e6ac77')
    for i,(g,b) in enumerate(zip(generation,relax)):axes[2].text(i,g+b+5,f'{g+b:.1f}s',ha='center')
    axes[2].set(title='Estimated cohort cost: all24 attempts',ylabel='Seconds (model loading / queue excluded)',ylim=(0,max(g+b for g,b in zip(generation,relax))*1.15))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('GSS12 / 24 paired CSP paths: 800-round cutoff versus full1000, after identical relaxation')
    for suffix in ('png','svg'):fig.savefig(root/f'comparison.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);root=p.parse_args().run
    draw(root,json.loads((root/'report.json').read_text()),json.loads((root/'complete.json').read_text()))
