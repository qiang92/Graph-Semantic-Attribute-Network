"""Paper fixed-prediction analyses; no signal encoder is retrained here."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import train_test_split

from utils.run_llm_gcn_zsl import normalize_rows
from utils.run_revision_component_benchmark import predict_nb, entropy_weights, match
from utils.run_revision_reliability_audit import inverse_weights, class_conditional_predict, mahalanobis_predict
from utils.run_tpts_llm_gcn_zsl import load_tpts, make_split, ATTR_MATRIX, predict_attributes, match_attributes, evaluate


def reliability(root):
    raw = np.load('data/TEP_attribute_matrix.npy').astype(np.float64)
    sem = normalize_rows(np.load('data/semantic_cache/tep_fault_semantics_qwen3_embedding.npy').astype(np.float32))
    attr = normalize_rows(raw.astype(np.float32))
    relations = .5*(attr@attr.T) + .5*(sem@sem.T)
    rows, correlations = [], []
    for folder in sorted(root.glob('seed*')):
        seed=int(folder.name[4:])
        graph_attr=np.load(folder/'graph_enhanced_attributes.npy').astype(np.float64)
        for task in range(1,5):
            with np.load(folder/f'gsan_artifacts_task{task}.npz') as data:
                labels=data['unseen_zero'].astype(np.int64)
                truth=data['test_y'].astype(np.int64)
                predicted=data['pred_attr'].astype(np.float64)
                pred_val=data['calibration_pred_val'].astype(np.float64)
                true_val=data['calibration_true_val'].astype(np.float64)
            mse=((pred_val-true_val)**2).mean(0)
            unseen_mse=((predicted-graph_attr[truth])**2).mean(0)
            correlations.append({'seed':seed,'task':task,'pearson':float(pearsonr(mse,unseen_mse).statistic)})
            prototypes=graph_attr[labels]
            validation_class=np.argmin(((true_val[:,None,:]-graph_attr[None,:,:])**2).sum(2),axis=1)
            seen=np.unique(validation_class)
            class_mse=np.stack([((pred_val[validation_class==label]-true_val[validation_class==label])**2).mean(0) for label in seen])
            conditional=[]
            for label in labels:
                relation=np.maximum(relations[label,seen],0.)
                relation/=relation.sum()+1e-12
                conditional.append(inverse_weights(relation@class_mse))
            precision=LedoitWolf().fit(pred_val-true_val).precision_
            predictions={
                'Uniform':match(predicted,prototypes,labels,np.ones(raw.shape[1])),
                'Inverse-MSE':match(predicted,prototypes,labels,inverse_weights(mse)),
                'Relation-conditioned MSE':class_conditional_predict(predicted,prototypes,labels,np.stack(conditional)),
                'Ledoit-Wolf Mahalanobis':mahalanobis_predict(predicted,prototypes,labels,precision)}
            for method,pred in predictions.items():
                rows.append({'seed':seed,'task':task,'method':method,'accuracy':float(100*np.mean(pred==truth))})
    return pd.DataFrame(rows),pd.DataFrame(correlations)


def entropy(root):
    attr=np.load('data/TEP_attribute_matrix.npy').astype(np.float32)
    rows=[]
    for folder in sorted(root.glob('seed*')):
        seed=int(folder.name[4:])
        for task in range(1,5):
            with np.load(folder/f'sce_artifacts_task{task}.npz') as data:
                train=data['train_feature'].astype(np.float64)
                targets=data['train_attr'].astype(np.float64)
                train_y=data['train_y'].astype(np.int64)
                test=data['feature_test'].astype(np.float64)
                truth=data['test_y'].astype(np.int64)
                labels=data['unseen_zero'].astype(np.int64)
            fit,val=train_test_split(np.arange(len(train)),test_size=.2,random_state=seed,stratify=train_y)
            hard=predict_nb(train,targets,test,probabilistic=False)
            probabilities=predict_nb(train[fit],targets[fit],train[val],probabilistic=True)
            for method,weights in [('Uniform',np.ones(attr.shape[1])),('Entropy',entropy_weights(probabilities))]:
                pred=match(hard,attr[labels],labels,weights)
                rows.append({'seed':seed,'task':task,'method':method,'accuracy':float(100*np.mean(pred==truth))})
    return pd.DataFrame(rows),pd.DataFrame()


def tpts_raw(path):
    xs,ys=load_tpts(path)
    rows=[]
    for a,b in itertools.combinations(range(1,6),2):
        train,train_y,targets,test,truth,prototypes,seen,unseen=make_split(xs,ys,[a,b])
        predicted=predict_attributes(train,targets,test,'NB',42)
        pred=match_attributes(predicted,prototypes,unseen,np.ones(4,dtype=np.float32))
        sample,macro,_=evaluate(truth,pred,unseen)
        rows.append({'seed':42,'task':f'{a},{b}','method':'Raw NB-Attr','accuracy':macro})
    return pd.DataFrame(rows),pd.DataFrame()


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('analysis',choices=['reliability','entropy','tpts_raw'])
    cli.add_argument('--run-dir',type=Path,default=Path('reference_results/tep_predictions'))
    cli.add_argument('--tpts-csv',type=Path,default=Path('data/classData.csv'))
    cli.add_argument('--output',required=True,type=Path)
    args=cli.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    detail,corr=tpts_raw(args.tpts_csv) if args.analysis=='tpts_raw' else globals()[args.analysis](args.run_dir)
    if detail.empty:
        raise ValueError('No seed/task artifacts found in the supplied directory.')
    detail.to_csv(args.output/'task_results.csv',index=False)
    per_run=detail.groupby(['method','seed'],as_index=False).accuracy.mean()
    per_run.to_csv(args.output/'run_results.csv',index=False)
    summary=per_run.groupby('method').accuracy.agg(['mean','std','count'])
    summary.to_csv(args.output/'summary.csv')
    if not corr.empty:
        corr.to_csv(args.output/'correlation_tasks.csv',index=False)
        corr.groupby('task').pearson.agg(['mean','std']).to_csv(args.output/'correlation_summary.csv')
    print(summary.to_string())


if __name__=='__main__':
    main()
