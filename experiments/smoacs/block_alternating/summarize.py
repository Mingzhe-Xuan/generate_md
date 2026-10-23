"""Report small-sample feasibility without treating budgets as convergence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

JOINT={'jointly_converged','jointly_constrained_or_hinge_stationary'}


def summarize(data,complete,gradient_proof):
    ids=data['protocol']['ids'];rows=data['candidates']
    if len(ids)!=4 or len(set(ids))!=4:raise ValueError('Expected four predetermined pilot IDs.')
    if len({r['candidate_id'] for r in rows})!=len(rows):raise ValueError('Duplicate candidate IDs.')
    if {r['candidate_id'] for r in rows}-set(ids):raise ValueError('Unexpected candidate IDs.')
    joint=[]
    for row in rows:
        if row['status'] not in JOINT:continue
        b=row['b_result']
        valid=(row['a_report']['converged'] and row['a_report']['projected_gradient']<=.001
            and b['status']=='converged' and b['max_filter_force']<=.01
            and row['b_endpoint_a_projected_gradient']<=.001 and row['stable_rounds']>=3
            and row['a_geometry_rms']<=.001 and row['a_cell_relative']<=1e-4
            and row['b_geometry_rms']<=.001 and row['b_cell_relative']<=1e-4
            and row['feedback_check']['passed'])
        if row['status']=='jointly_converged':valid=valid and row['b_endpoint_a_autograd_gradient']<=.001
        if not valid:raise ValueError('Declared joint candidate fails independent residual checks: '+row['candidate_id'])
        joint.append(row['candidate_id'])
    finished=bool(complete.get('completed') and len(rows)==4 and all(r['status']!='active' for r in rows))
    return dict(processing_completed=finished,denominator=4,observed_records=len(rows),
        statuses=dict(Counter(r['status'] for r in rows)),joint_confirmed=len(joint),joint_ids=joint,
        raw_autograd_joint=sum(r['status']=='jointly_converged' for r in rows),
        last_A_stationary=sum(r.get('a_report',{}).get('converged',False) for r in rows),
        last_B_converged=sum(r.get('b_result',{}).get('status')=='converged' for r in rows),
        gradient_probe_passed=gradient_proof.get('corrected_all_pass',False),candidates=rows,
        conclusion=('Joint stationarity observed within declared numerical tolerances; this does not establish thermodynamic stability.'
            if joint else 'No joint stationary candidate confirmed. A solver/relaxation failures or finite budgets do not prove a common stationary point cannot exist.'),
        protocol=data['protocol'],note='Last B may precede an unconverged later A phase. Only joint-status rows satisfy both current phases. '
            'Fixed-T L-BFGS-B and corrected CPU derivatives differ from the original baseline; no isolated alternation-effect claim or DFT/SUN claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--gradient-probe',type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((args.run/'protocol.json').read_text())
    data=json.loads((args.run/'results.json').read_text()) if (args.run/'results.json').exists() else dict(protocol=protocol,candidates=[])
    complete=json.loads((args.run/'complete.json').read_text()) if (args.run/'complete.json').exists() else {}
    proof=json.loads(args.gradient_probe.read_text())
    report=summarize(data,complete,proof)
    for row in report['candidates']:
        if row['status'] not in JOINT:continue
        b=row['b_result']
        for path,expected in [(row['a_source'],b['input_sha256']),(b['relaxed_path'],b['relaxed_sha256'])]:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Joint geometry hash mismatch.')
    (args.run/'feasibility-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Small-sample block A/B feasibility','',
        'Processing completed: '+str(report['processing_completed'])+'. Joint confirmed: '+str(report['joint_confirmed'])+'/4.',
        '',report['conclusion'],'',
        '| Candidate | Status | A phases | A residual | B-end A residual | Last B force |',
        '|---|---|---:|---:|---:|---:|']
    for row in report['candidates']:
        values=[row['candidate_id'],row['status'],row.get('a_phases'),row.get('a_report',{}).get('projected_gradient'),
                row.get('b_endpoint_a_projected_gradient'),row.get('b_result',{}).get('max_filter_force')]
        lines.append('| '+' | '.join(str(v) for v in values)+' |')
    lines+=['',report['note'],'','Gradient probe passed: '+str(report['gradient_probe_passed']), '']
    (args.run/'feasibility-report.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in report.items() if k not in ('candidates','protocol')}),flush=True)


if __name__=='__main__':main()
