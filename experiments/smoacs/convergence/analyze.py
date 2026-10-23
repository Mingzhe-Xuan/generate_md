"""Plot original CPU206 property histories and diagnose late-stage changes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from diagnostics import objective, window_changes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    config=json.loads(args.config.read_text(encoding='utf8'))
    with args.csv.open(encoding='utf8') as stream:
        originals={r['original_fnames']:r for r in csv.DictReader(stream)}
    with np.load(args.npz,allow_pickle=False) as z:
        gap=z['gap_history'].astype(float)
        ef=z['ef_history'].astype(float)
        ids=z['original_fnames'].tolist()
        onehot_gap=z['bandgap_onehot'].astype(float)
        onehot_ef=z['ef_onehot'].astype(float)
        dist_gap=z['bandgap_dist'].astype(float)
        dist_ef=z['ef_dist'].astype(float)
    if gap.shape != (config['num_steps'],config['num_candidate']) or len(set(ids)) != len(ids) or set(ids) != set(originals):
        raise ValueError('History shape or candidate IDs do not match the original experiment.')
    if max(abs(onehot_gap[i]-float(originals[k]['bandgap_onehot'])) for i,k in enumerate(ids)) > 1e-5:
        raise ValueError('NPZ does not match archived CSV.')
    if max(abs(onehot_ef[i]-float(originals[k]['ef_onehot'])) for i,k in enumerate(ids)) > 1e-5:
        raise ValueError('NPZ does not match archived CSV formation energies.')
    loss=objective(gap,ef,config['target_bandgap'],config['bandgap_margin'])
    args.output.mkdir(parents=True,exist_ok=False)
    wide=(gap>=2.3)&(gap<=2.7)&(ef<-.5)
    narrow=(np.abs(gap-2.5)<=.04)&(ef<-.5)
    series=[]
    for i in range(len(gap)):
        series.append(dict(completed_updates=i, forward_record=i+1,
                           objective_mean=float(loss[i].mean()),objective_median=float(np.median(loss[i])),
                           gap_absolute_error_mean=float(np.abs(gap[i]-2.5).mean()),ef_mean=float(ef[i].mean()),
                           property_only_wide_pass=int(wide[i].sum()),property_only_training_margin_pass=int(narrow[i].sum()),
                           temperature=.01-(.01-.0001)*i/config['num_steps'],
                           lr_fraction=.5*(1+np.cos(np.pi*i/config['num_steps']))))
    changes={}
    for start in (149,179,189):
        changes[str(start)+'_to_199']={name:window_changes(values,start,199)
                                    for name,values in [('objective',loss),('gap',gap),
                                                        ('gap_absolute_error',np.abs(gap-2.5)),('ef',ef)]}
    candidates=[]
    for i,k in enumerate(ids):
        candidates.append(dict(candidate_id=k,objective_delta_179_199=float(loss[199,i]-loss[179,i]),
            gap_delta_179_199=float(gap[199,i]-gap[179,i]),ef_delta_179_199=float(ef[199,i]-ef[179,i]),
            objective_span_180_199=float(np.ptp(loss[180:200,i])),
            gap_span_180_199=float(np.ptp(gap[180:200,i])),ef_span_180_199=float(np.ptp(ef[180:200,i]))))
    summary=dict(candidate_count=len(ids),history_shape=list(gap.shape),
        npz_sha256=hashlib.sha256(args.npz.read_bytes()).hexdigest(),
        reconstructed_objective='max(abs(gap-2.5)-0.04,0)+predicted formation energy; coefficient1, noEfclip',
        history_semantics='Row i is pre-update prediction at completed_updates=i, i=0..199. Final evaluation uses1e-8 temperature; do not append it to the training-temperature trajectory.',
        windows=changes,selected_records=[series[i] for i in (0,99,149,179,189,199)],
        last_record=dict(gap_mean_absolute_error=float(np.abs(gap[-1]-2.5).mean()),
                         ef_mean=float(ef[-1].mean()),objective_mean=float(loss[-1].mean())),
        final_discrete=dict(property_only_pass=int(((onehot_gap>=2.3)&(onehot_gap<=2.7)&(onehot_ef<-.5)).sum()),
                            joint_with_archived_validity=int(sum(2.3<=onehot_gap[i]<=2.7 and onehot_ef[i]<-.5 and originals[k]['valid_structure']=='True' for i,k in enumerate(ids)))),
        final_distribution=dict(gap_mean_absolute_error=float(np.abs(dist_gap-2.5).mean()),
                                ef_mean=float(dist_ef.mean()),objective_mean=float(objective(dist_gap,dist_ef).mean())),
        limitations=['No saved gradient norm, optimizer state, or geometric history.',
                     'Learning rate is deliberately annealed to zero; flat history alone does not prove stationarity.',
                     'Atomic temperature changes across history; last-stage changes mix parameter updates and temperature effects.',
                     'Per-step success is property-only, without per-step neutrality/distance checks.'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,rows in [('history.csv',series),('candidates.csv',candidates)]:
        with (args.output/name).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plots(args.output,gap,ef,loss,wide,narrow,series,summary)
    print(json.dumps(summary,indent=2))


def plots(output,gap,ef,loss,wide,narrow,series,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if any(f.name=='Microsoft YaHei' for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus']=False
    x=np.arange(len(gap))
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,values,title,ylabel in [(axes[0,0],loss,'A  原训练目标函数','目标函数值'),
                                   (axes[0,1],np.abs(gap-2.5),'B  带隙目标绝对误差','eV'),
                                   (axes[1,0],ef,'C  预测形成能','eV/atom')]:
        ax.fill_between(x,np.quantile(values,.25,axis=1),np.quantile(values,.75,axis=1),alpha=.18,color='#2563a6',label='P25–P75')
        ax.plot(x,np.mean(values,axis=1),color='#2563a6',label='均值')
        ax.plot(x,np.median(values,axis=1),color='#178878',linestyle='--',label='中位数')
        ax.set(title=title,xlabel='已完成的梯度更新数（当前更新前预测）',ylabel=ylabel)
        ax.axvspan(179,199,color='#e39527',alpha=.12)
        ax.legend(fontsize=9)
    ax=axes[1,1]
    ax.plot(x,wide.sum(axis=1),label='带隙±0.2及Ef<−0.5',color='#2563a6')
    ax.plot(x,narrow.sum(axis=1),label='带隙±0.04及Ef<−0.5',color='#178878')
    ax.scatter([200],[summary['final_discrete']['property_only_pass']],marker='*',s=140,color='#bc5329',label='最终one-hot性质通过')
    ax.set(title='D  性质达标计数（未含逐步结构有效性）',xlabel='已完成的更新数',ylabel='候选数 / 128',ylim=(0,128))
    ax.legend(fontsize=9)
    fig.suptitle('CPU206：200步内是否已进入稳定平台？',fontsize=16)
    fig.text(.5,-.025,'阴影标示末20次记录区间；温度持续变化，学习率退火至0。平台不等于梯度已为0。',ha='center')
    for ext in ('png','svg'):fig.savefig(output/('convergence.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
    axes[0].hist(loss[199]-loss[179],bins='fd',color='#2563a6')
    for value in (-.01,.01):axes[0].axvline(value,color='#bc5329',linestyle='--')
    axes[0].set(title='末20次更新的目标函数变化',xlabel='第199次完成更新状态 − 第179次状态',ylabel='候选数')
    axes[1].plot(x,[r['lr_fraction'] for r in series],label='学习率 / 初始学习率')
    axes[1].plot(x,np.array([r['temperature'] for r in series])/.01,label='温度 / 初始温度')
    axes[1].set(title='调度器在200步终止前的变化',xlabel='已完成更新数',ylabel='相对初值')
    axes[1].legend()
    for ext in ('png','svg'):fig.savefig(output/('tail_and_schedule.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
