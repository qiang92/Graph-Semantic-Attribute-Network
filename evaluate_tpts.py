"""Evaluate the ten primary-configuration TPTS models from seed 42."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tpts_inference import Predictor
from utils.run_tpts_llm_gcn_zsl import load_tpts


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data',required=True,type=Path)
    cli.add_argument('--models',type=Path,default=Path(__file__).resolve().parent/'models/tpts')
    cli.add_argument('--output',required=True,type=Path)
    args=cli.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new output directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    xs,ys=load_tpts(args.data)
    rows=[]
    for folder in sorted(args.models.glob('pair_*')):
        model=Predictor(folder)
        labels=model.metadata['candidate_indices']
        raw=np.vstack([xs[i] for i in labels])
        truth=np.concatenate([ys[i] for i in labels])
        result=model.predict(raw,batch_size=257)
        pred=result['class_indices']
        macro=float(np.mean([100*np.mean(pred[truth==label]==label) for label in labels]))
        sample=float(100*np.mean(pred==truth))
        rows.append({'pair':folder.name,'seed':model.metadata['seed'],
                     'macro_accuracy_percent':macro,'sample_accuracy_percent':sample,
                     'reference_macro_percent':model.metadata['macro_accuracy_percent']})
        np.savez_compressed(args.output/(folder.name+'.npz'),truth=truth,**result)
    if len(rows)!=10:
        raise ValueError(f'Expected ten fault pairs, found {len(rows)}')
    frame=pd.DataFrame(rows)
    frame.to_csv(args.output/'task_results.csv',index=False)
    result={'seed':42,'splits':10,'mean_macro_accuracy_percent':float(frame.macro_accuracy_percent.mean()),
            'all_pairs_match_reference':bool(np.allclose(frame.macro_accuracy_percent,frame.reference_macro_percent,rtol=0,atol=1e-10)),
            'paper_five_run_mean_percent':73.05639642968329,
            'scope':'One primary-configuration run, not the five-run statistic.'}
    (args.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
