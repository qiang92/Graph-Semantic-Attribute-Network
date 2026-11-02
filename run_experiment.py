"""Run the accepted-paper configurations and aggregate within-run task means."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def arguments(options):
    output = []
    for key, value in options.items():
        if value is None:
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                output.append(flag)
            elif key == 'use_graph':
                output.append('--no-use-graph')
        elif isinstance(value, list):
            output.extend([flag, *map(str, value)])
        else:
            output.extend([flag, str(value)])
    return output


def aggregate(rows):
    frame = pd.DataFrame(rows)
    # Tasks have equal weight within a run, regardless of sample count.
    per_run = frame.groupby('seed', sort=False).accuracy.mean()
    return {'runs': len(per_run), 'mean_percent': float(per_run.mean()),
            'sample_std_percent': float(per_run.std(ddof=1)) if len(per_run)>1 else None,
            'run_results': [{'seed': int(seed), 'accuracy_percent': float(value)} for seed,value in per_run.items()]}


def main():
    registry = json.loads((ROOT/'configs/experiments.json').read_text())
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('experiment', choices=registry)
    cli.add_argument('--seeds', type=int, nargs='+')
    cli.add_argument('--data-dir', type=Path, default=ROOT/'data/TEP_data')
    cli.add_argument('--tpts-csv', type=Path, default=ROOT/'data/classData.csv')
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--epochs', type=int, help='Override only for development; paper budget remains in configs.')
    cli.add_argument('--tep-tasks', type=int, nargs='+', help='Subset for development; TEP main/prototype runner only.')
    cli.add_argument('--tpts-pairs', nargs='+', help='e.g. 1,2 1,3; omitted means all ten pairs')
    args = cli.parse_args()
    recipe = registry[args.experiment]
    seeds = args.seeds if args.seeds is not None else recipe['seeds']
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory so old cells cannot enter the aggregate.')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    task_rows, jobs, generalized_rows = [], [], []
    for index, seed in enumerate(seeds):
        options = dict(recipe['options'])
        options['seed'] = seed
        if args.epochs is not None:
            options['epochs'] = args.epochs
        if recipe['dataset']=='tep':
            options['data_path'] = str(args.data_dir.resolve()/'*.dat')
            if args.tep_tasks is not None:
                options['tasks'] = args.tep_tasks
        else:
            options['data_path'] = str(args.tpts_csv.resolve())
        if recipe.get('random_graph_per_seed'):
            graph_index = recipe['seeds'].index(seed)
            options['fixed_adjacency'] = f'data/graphs/Random20_{graph_index}.npy'
        pairs = args.tpts_pairs if args.tpts_pairs else [f'{a},{b}' for a,b in itertools.combinations(range(1,6),2)]
        for pair in pairs if recipe['dataset']=='tpts' else [None]:
            jobdir = output/f'seed{seed}' if pair is None else output/f'seed{seed}'/('pair_'+pair.replace(',','_'))
            options['out_dir'] = str(jobdir)
            if pair is not None:
                options['unseen']=pair
            command = [sys.executable, recipe['script'], *arguments(options)]
            jobdir.mkdir(parents=True,exist_ok=True)
            jobs.append({'seed':seed,'pair':pair,'command':command})
            (output/'commands.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
            with (jobdir/'training.log').open('w',encoding='utf-8') as log:
                result = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Training failed; see {jobdir / "training.log"}')
            if recipe.get('result_type')=='gzsfd':
                generalized_rows.extend(pd.read_csv(jobdir/'gzsfd_task_results.csv').to_dict('records'))
                continue
            if recipe['dataset']=='tpts':
                payload=json.loads((jobdir/'summary.json').read_text())
                task_rows.append({'seed':seed,'task':pair,'accuracy':payload['per_class_top1']})
            elif (jobdir/'gsan_summary.json').exists():
                payload=json.loads((jobdir/'gsan_summary.json').read_text())
                task_ids=pd.read_csv(jobdir/'gsan_details.csv').task
                for task,accuracy in zip(task_ids,payload['task_accuracy']):
                    task_rows.append({'seed':seed,'task':task,'accuracy':100*accuracy})
            elif options.get('compare_reliability'):
                for weighting in ['uniform','inverse_mse']:
                    payload=json.loads((Path(str(jobdir)+'_'+weighting)/'gsan_summary.json').read_text())
                    task_ids=pd.read_csv(Path(str(jobdir)+'_'+weighting)/'gsan_details.csv').task
                    for task,accuracy in zip(task_ids,payload['task_accuracy']):
                        task_rows.append({'seed':seed,'task':task,'accuracy':100*accuracy,'weighting':weighting})
            elif (jobdir/'sce_clean_summary.json').exists():
                payload=json.loads((jobdir/'sce_clean_summary.json').read_text())
                for task,accuracy in enumerate(payload['task_accuracy'],start=1):
                    task_rows.append({'seed':seed,'task':task,'accuracy':100*accuracy})
            elif (jobdir/'llm_gcn_zsl_details.csv').exists():
                for row in pd.read_csv(jobdir/'llm_gcn_zsl_details.csv').to_dict('records'):
                    task_rows.append({'seed':seed,'task':row['task'],'accuracy':row['per_class_top1']})
            elif (jobdir/'sce_llm_gcn_tep_summary.json').exists():
                payload=json.loads((jobdir/'sce_llm_gcn_tep_summary.json').read_text())
                for task,accuracy in enumerate(payload['task_accuracy'],start=1):
                    task_rows.append({'seed':seed,'task':task,'accuracy':100*accuracy})
            print(f'Completed {args.experiment}: seed={seed} pair={pair}',flush=True)
    if task_rows:
        pd.DataFrame(task_rows).to_csv(output/'task_results.csv',index=False)
        summary=aggregate(task_rows) if 'weighting' not in task_rows[0] else {
            'weighting_results':{weighting:aggregate([r for r in task_rows if r['weighting']==weighting])
                                for weighting in ['uniform','inverse_mse']}}
        summary.update(experiment=args.experiment,development_budget=args.epochs is not None,
                       task_subset=args.tep_tasks,tpts_pair_subset=args.tpts_pairs,
                       paper_reference=recipe.get('paper_accuracy_percent'))
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))
    if generalized_rows:
        frame=pd.DataFrame(generalized_rows)
        metrics=['seen_accuracy','unseen_accuracy','harmonic_mean']
        frame.to_csv(output/'gzsfd_task_results.csv',index=False)
        per_run=frame.groupby(['method','seed'],as_index=False)[metrics].mean()
        per_run.to_csv(output/'gzsfd_run_results.csv',index=False)
        per_run.groupby('method')[metrics].agg(['mean','std']).to_csv(output/'gzsfd_summary.csv')


if __name__=='__main__':
    main()
