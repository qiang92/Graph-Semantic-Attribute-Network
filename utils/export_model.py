"""Save a selected NB diagnosis model without changing the training trajectory."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from tpts_inference import predict_attributes


def export_selected(local, dataset):
    args, model, state = local['args'], local['model'], local['state']
    # Other exploratory classifiers remain trainable; this inference format is NB.
    if dataset == 'TEP' and (args.attr_predictor != 'nb_binary' or args.adaptive_betas or args.balanced_assignment):
        return
    if dataset == 'TPTS' and (args.attr_model != 'NB' or args.eval_mode != 'nb_attr'):
        return
    task = f"task_{local['task_id']}" if dataset == 'TEP' else 'pair_' + args.unseen.replace(',', '_')
    out = Path(args.out_dir) / 'models' / local['weighting'] / task
    out.mkdir(parents=True, exist_ok=True)
    features = local['f_train'] if dataset == 'TEP' else local['train_feat']
    targets = local['train_attr_fit'] if dataset == 'TEP' else local['train_attr']
    expected = state['pred_attr'] if dataset == 'TEP' else local['pred_attr']
    arrays, info, heads = {}, [], []
    for j in range(targets.shape[1]):
        target = (targets[:, j] >= .5).astype(np.int64) if dataset == 'TEP' else targets[:, j]
        if target.max() == target.min():
            head = {'constant': float(target[0])}
            info.append(head)
        else:
            clf = GaussianNB().fit(features, target)
            head = {'classes': clf.classes_, 'theta': clf.theta_, 'var': clf.var_, 'prior': clf.class_prior_}
            info.append({'type': 'GaussianNB', 'target_threshold': .5 if dataset == 'TEP' else None})
            arrays.update({f'head_{j}_{key}': value for key, value in head.items()})
        heads.append(head)
    probe = local['f_test'] if dataset == 'TEP' else local['test_feat']
    np.testing.assert_array_equal(predict_attributes(probe, heads), expected)
    prefixes = ('x1.', 'x2.', 'bn_hidden_x.', 'ln_hidden.') if dataset == 'TEP' else ('signal.',)
    torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                if key.startswith(prefixes)}, out / 'encoder.pt')
    labels = local['unseen_zero']
    weights = state['calibration']['weights'] if dataset == 'TEP' else state['attribute_weights']
    arrays.update(prototypes=local['test_attr_m'], weights=weights, candidate_indices=labels)
    if dataset == 'TEP':
        cfg = local['cfg']
        encoder = {'input_dim': local['train_x'].shape[1], 'hidden': cfg['dim'][0], 'latent': cfg['dim'][1]}
        preprocessing = {'kind': 'per_record_trailing_SMA', 'window': cfg.get('sma_window', 3), 'standardization': 'none'}
        class_names = {str(i): f'F{i+1}' for i in range(15)}
        arrays.update(class_graph=local['adj'], all_prototypes=local['graph_attr'])
        accuracy = 100 * state['acc']
    else:
        from utils.run_tpts_llm_gcn_zsl import CLASS_ORDER, ATTR_MATRIX
        scaler = StandardScaler().fit(np.vstack([local['x_all'][i] for i in local['seen_zero']]))
        arrays.update(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, scaler_var=scaler.var_)
        encoder = {'input_dim': 6, 'hidden': args.hidden, 'latent': args.latent}
        preprocessing = {'kind': 'seen_fitted_StandardScaler', 'input_columns': ['Ia','Ib','Ic','Va','Vb','Vc']}
        class_names = dict(enumerate(CLASS_ORDER))
        arrays.update(class_graph=local['adj'], all_prototypes=ATTR_MATRIX)
        accuracy = state['sample_acc']
    np.savez_compressed(out / 'inference_arrays.npz', **arrays)
    metadata = {'dataset': dataset, 'task': task, 'seed': args.seed, 'selected_epoch': local['epoch']+1,
                'selection': 'highest unseen-test sample accuracy' if dataset == 'TEP' else 'highest unseen-test macro accuracy',
                'encoder': encoder, 'preprocessing': preprocessing, 'attribute_heads': info,
                'class_names': class_names, 'candidate_indices': labels.tolist(),
                'sample_accuracy_percent': accuracy,
                'macro_accuracy_percent': accuracy if dataset == 'TEP' else state['per_class_top1']}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, out / 'training_model.pt')
