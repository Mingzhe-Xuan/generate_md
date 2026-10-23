"""Describe CPU206 Crystalformer and candidate MatterSim iteration costs."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from measurements import crystalformer_steps, mattersim_steps, describe


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(output, cf, records, launch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    if any(f.name == 'Microsoft YaHei' for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams.update({'axes.unicode_minus':False, 'font.size':11,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.fonttype':'none'})
    converged = [r for r in records if r['status'] == 'converged']
    capped = [r for r in records if r['status'] == 'unconverged']
    colors = ('#2563a6', '#178878', '#e39527')
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax[0,0].hist([launch['steps']]*len(records), bins=[199.5,200.5],color=colors[0],rwidth=.55)
    ax[0,0].set(xlabel='每个候选的更新步数', ylabel='候选数', title='A  Crystalformer：固定 200 步（128 个候选）',xticks=[200],xlim=(190,210))
    ax[0,0].text(.04,.94,'同一批次共享 200 次更新；不是独立收敛步数',transform=ax[0,0].transAxes,va='top',fontsize=10)
    bins = np.arange(0,426,25)
    ax[0,1].hist([r['logged_updates'] for r in converged],bins=bins,color=colors[1],alpha=.85,label='收敛（n=115）')
    ax[0,1].hist([r['logged_updates'] for r in capped],bins=bins,histtype='step',color=colors[2],linewidth=2,label='未收敛、达到上限（n=4）')
    ax[0,1].set(xlabel='每个结构的 BFGS 更新步数（不含 step 0）',ylabel='结构数',title='B  MatterSim：实际步数分布')
    ax[0,1].legend(fontsize=9)
    ax[1,0].hist([r['seconds'] for r in cf],bins=[11.5,12.5,13.5],rwidth=.8,color=colors[0])
    ax[1,0].set(xlabel='秒 / 批量更新步（每批 128 个候选）',ylabel='更新步数',title='C  Crystalformer：CPU 批量更新耗时',xticks=[12,13])
    rates = [r['mean_seconds_per_step'] for r in converged if r['mean_seconds_per_step'] is not None]
    ax[1,1].hist(rates,bins='fd',color=colors[1],alpha=.85)
    ax[1,1].set(xlabel='每结构平均秒 / BFGS 步',ylabel='收敛结构数',title='D  MatterSim：RTX 5090 上的平均步耗时')
    ax[1,1].text(.98,.95,'0 值来自秒级日志量化',ha='right',va='top',transform=ax[1,1].transAxes,fontsize=10)
    fig.suptitle('同一批生成候选的优化成本：Crystalformer 与 MatterSim',fontsize=16)
    fig.text(.5,-.035,'时间从秒级日志重建，均为近似值；D 对每个结构先取平均。CPU 批量更新与 GPU 单结构步不能直接作加速比。\n9 个失败样本只记录了部分步数，不计入收敛统计；4 个达到上限的未收敛样本以橙线单列。',ha='center',fontsize=10)
    for suffix in ('png','svg'):
        fig.savefig(output/('histograms.'+suffix),dpi=190,bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
    intervals = [v['seconds'] for r in records if r['status'] in ('converged','unconverged') for v in r['intervals']]
    max_interval = max(intervals)
    ax[0].hist(intervals,bins=np.arange(-.5,max_interval+1.5,1),color=colors[1],rwidth=.8)
    ax[0].set(xlabel='相邻 BFGS 日志时间差（秒）',ylabel='记录区间数',title='原始单步间隔：秒级量化，0 不代表实际零耗时')
    ax[0].set_xticks(range(max_interval+1))
    ax[1].hist([r['logged_span_seconds'] for r in converged],bins='fd',color=colors[1])
    ax[1].set(xlabel='step 0 到最终步的记录时间跨度（秒）',ylabel='收敛结构数',title='每结构 BFGS 阶段耗时（不含模型加载）')
    for suffix in ('png','svg'):
        fig.savefig(output/('timing_details.'+suffix),dpi=190,bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crystalformer', type=Path, required=True, help='Archived evidence-206 directory')
    parser.add_argument('--mattersim', type=Path, required=True, help='Archived candidate relaxation directory')
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding='utf8'))
    launch_path = args.crystalformer/'section21-Struct-AtomicMask-grad-206-launch.json'
    launch = json.loads(launch_path.read_text(encoding='utf8'))['arguments']
    if launch['candidates'] != inventory['denominator'] or launch['batch_size'] != inventory['denominator']:
        raise ValueError('Expected the same complete128-candidate single batch.')
    progress = args.crystalformer/'section21-206.err'
    cf = crystalformer_steps(progress.read_text(encoding='utf8'), launch['steps'])
    records, hashes = [], {str(progress):digest(progress), str(launch_path):digest(launch_path), str(args.inventory):digest(args.inventory)}
    seen = set()
    for candidate in inventory['candidates']:
        identifier = candidate['candidate_id']
        if identifier in seen:
            raise ValueError('Duplicate inventory ID.')
        seen.add(identifier)
        folder = args.mattersim/Path(identifier).stem
        result_path = folder/'result.json'
        result = json.loads(result_path.read_text(encoding='utf8'))
        if result['input_sha256'] != candidate['sha256']:
            raise ValueError('Timing and candidate inventory provenance differ.')
        log = folder/'relax.log'
        complete = result['status'] in ('converged','unconverged')
        values = mattersim_steps(log.read_text(encoding='utf8') if log.exists() else '',
                                result['optimizer_steps'] if complete else None)
        records.append(dict(candidate_id=identifier,status=result['status'],num_atoms=candidate['num_atoms'],**values))
        hashes[str(result_path)] = digest(result_path)
        if log.exists():
            hashes[str(log)] = digest(log)
    if len(records) != inventory['denominator']:
        raise ValueError('Candidate denominator mismatch.')
    summary = dict(statuses=dict(Counter(r['status'] for r in records)),statistics={},
                   crystalformer_loop_seconds=cf[-1]['elapsed_seconds'],
                   clock_resolution_seconds=1, quantile_method='linear interpolation at (n-1)*q',
                   units_note='Crystalformer is one CPU batch of128; MatterSim is one structure per GPU optimizer. '
                   'Logged intervals are integer-second approximations. Zero interval is quantization, not zero work. '
                   'Per-structure mean=logged span / completed updates. '
                   'Step0, model loading, initial evaluation and final export are excluded. '
                   'Fixed200 and censored400 are not demonstrated convergence requirements.',
                   input_sha256=hashes)
    groups = {
        'crystalformer_steps_per_candidate':([launch['steps']]*len(records),'updates'),
        'crystalformer_seconds_per_batch_update':([r['seconds'] for r in cf],'s/batch-update'),
    }
    for status in ('converged','unconverged','failed'):
        group = [r for r in records if r['status'] == status]
        groups['mattersim_'+status+'_logged_updates'] = ([r['logged_updates'] for r in group if r['logged_updates'] is not None],'updates')
        if status != 'failed':
            groups['mattersim_'+status+'_mean_seconds_per_update'] = ([r['mean_seconds_per_step'] for r in group if r['mean_seconds_per_step'] is not None],'s/update')
            groups['mattersim_'+status+'_logged_span_seconds'] = ([r['logged_span_seconds'] for r in group],'s')
            groups['mattersim_'+status+'_raw_interval_seconds'] = ([v['seconds'] for r in group for v in r['intervals']],'s/recorded-interval')
    for name,(values,unit) in groups.items():
        summary['statistics'][name] = dict(unit=unit,**describe(values))
    converged = [r for r in records if r['status'] == 'converged']
    summary['mattersim_converged_pooled_seconds_per_update'] = (
        sum(r['logged_span_seconds'] for r in converged)/sum(r['logged_updates'] for r in converged))
    summary['mattersim_zero_span_converged_ids'] = [r['candidate_id'] for r in converged if r['logged_span_seconds']==0]
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    write_csv(args.output/'statistics.csv',[dict(metric=k,**v) for k,v in summary['statistics'].items()])
    write_csv(args.output/'crystalformer_steps.csv',cf)
    write_csv(args.output/'mattersim_structures.csv',[{k:v for k,v in r.items() if k!='intervals'} for r in records])
    write_csv(args.output/'mattersim_intervals.csv',[dict(candidate_id=r['candidate_id'],status=r['status'],**v) for r in records for v in r['intervals']])
    figures(args.output,cf,records,launch)
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},indent=2))


if __name__ == '__main__':
    main()
