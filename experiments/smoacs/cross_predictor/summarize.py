"""Summarize paired Crystalformer/ALIGNN predictions and plot agreement."""
import argparse
import csv
import json
from pathlib import Path

from metrics import agreement,index_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=json.loads((args.input/'predictions.json').read_text())
    metadata=json.loads((args.input/'metadata.json').read_text())
    before=[r for r in rows if r['stage']=='before'];after=[r for r in rows if r['stage']=='after']
    ids=[r['candidate_id'] for r in before]
    index_rows(before,ids);index_rows(after,ids)
    if len(ids)!=metadata['denominator'] or len(metadata['controls'])<2:
        raise ValueError('Incomplete inventory or implementation controls.')
    if any(r['status']=='pending' for r in rows):raise ValueError('Partial inference output.')
    eligible=[r for r in after if r['status']!='not_eligible_relaxation']
    paired_ids={r['candidate_id'] for r in eligible}
    summary=dict(before_full=agreement(before),after_full=agreement(after),
        before_converged_pairs=agreement([r for r in before if r['candidate_id'] in paired_ids]),
        after_converged_pairs=agreement(eligible),metadata=metadata,
        failures=[dict(candidate_id=r['candidate_id'],stage=r['stage'],status=r['status'],error=r.get('error')) for r in rows if r['status']!='completed'],
        note='Differences/correlations compare predictors, not accuracy against ground truth. Unknown outcomes retain the original denominator. Retention counts only confirmed ALIGNN passes.')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    flat=[]
    for row in rows:
        r={k:v for k,v in row.items() if not isinstance(v,dict)}
        for model in ('crystalformer','alignn'):
            r.update({model+'_'+k+'_pass':v for k,v in row[model+'_flags'].items()})
        flat.append(r)
    with (args.output/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in flat))))
        writer.writeheader();writer.writerows(flat)
    lines=['# Crystalformer vs frozen ALIGNN', '',
           'Same geometry and target thresholds. Numerical errors are model-to-model differences, not ground-truth errors.', '']
    for key in ('before_full','after_converged_pairs'):
        group=summary[key]
        lines.extend(['## '+key+' (N='+str(group['denominator'])+')','',
            '| Criterion | Crystalformer pass | ALIGNN pass | Both pass | CF-only | ALIGNN-only | Unknown pair | CF-pass retention |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for metric,data in group['classification'].items():
            retained=data['confirmed_retention_of_crystalformer_pass']
            values=[data[k] for k in ('crystalformer_pass','alignn_pass','both_pass','crystalformer_only','alignn_only','unknown')]
            lines.append('| '+metric+' | '+' | '.join(map(str,values))+' | '+(f'{100*retained:.2f}%' if retained is not None else 'undefined')+' |')
        lines.extend(['','| Property | Known pairs | Bias ALIGNN-CF | MAE between predictors | RMSE | Pearson | Spearman |',
                      '|---|---:|---:|---:|---:|---:|---:|'])
        for metric,data in group['numerical'].items():
            lines.append('| '+metric+' | '+' | '.join(str(data.get(k)) for k in ('n','bias_alignn_minus_crystalformer','mae_between_predictors','rmse_between_predictors','pearson','spearman'))+' |')
        lines.append('')
    (args.output/'report.md').write_text('\n'.join(lines),encoding='utf8')
    plots(args.output,before,eligible)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('metadata','failures')},indent=2))


def plots(output,before,after):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    if any(f.name=='Microsoft YaHei' for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams.update({'axes.unicode_minus':False,'svg.fonttype':'none'})
    palette={'both_pass':'#178878','crystalformer_only':'#d18424','alignn_only':'#9155a3','neither_pass':'#8393a3'}
    labels={'both_pass':'两者均达标','crystalformer_only':'仅Crystalformer达标','alignn_only':'仅ALIGNN达标','neither_pass':'两者均未达标'}
    fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
    for col,(data,title) in enumerate(((before,'MatterSim前'),(after,'MatterSim后'))):
        completed=[r for r in data if r['status']=='completed']
        for i,prop in enumerate(('gap','eform')):
            ax=axes[i,col]
            for category,color in palette.items():
                selected=[]
                for r in completed:
                    a,b=r['crystalformer_flags'][prop],r['alignn_flags'][prop]
                    key='both_pass' if a and b else 'crystalformer_only' if a else 'alignn_only' if b else 'neither_pass'
                    if key==category:selected.append(r)
                ax.scatter([r['crystalformer_'+prop] for r in selected],[r['alignn_'+prop] for r in selected],
                           c=color,label=labels[category],s=28,alpha=.8)
            values=[r[m+'_'+prop] for r in completed for m in ('crystalformer','alignn')]
            lo,hi=(min(values),max(values)) if values else (0,1)
            pad=max((hi-lo)*.05,.1);lo-=pad;hi+=pad
            ax.plot([lo,hi],[lo,hi],color='#333333',linestyle='--',linewidth=1,label='完全一致')
            if prop=='gap':
                ax.axvspan(2.3,2.7,color='#e8be56',alpha=.12);ax.axhspan(2.3,2.7,color='#e8be56',alpha=.12)
            else:
                ax.axvline(-.5,color='#b85c38',linestyle=':');ax.axhline(-.5,color='#b85c38',linestyle=':')
            unit='带隙/eV' if prop=='gap' else '形成能/(eV/atom)'
            ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Crystalformer '+unit,ylabel='ALIGNN '+unit,
                   title=title+'：'+unit+'，有效配对n='+str(len(completed)))
            ax.legend(fontsize=8)
    fig.suptitle('同一结构的独立架构交叉预测',fontsize=16)
    for ext in ('png','svg'):fig.savefig(output/('agreement.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,prop in zip(axes,('gap','eform')):
        for data,label in ((before,'MatterSim前'),(after,'MatterSim后')):
            d=[r['alignn_'+prop]-r['crystalformer_'+prop] for r in data if r['status']=='completed']
            if d:ax.hist(d,bins=min(25,max(5,int(np.sqrt(len(d))))),alpha=.5,label=label)
        ax.axvline(0,color='black',linestyle='--')
        ax.set(xlabel='ALIGNN − Crystalformer（'+('eV' if prop=='gap' else 'eV/atom')+'）',ylabel='结构数',title='带隙差值' if prop=='gap' else '形成能差值')
        ax.legend()
    for ext in ('png','svg'):fig.savefig(output/('differences.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
