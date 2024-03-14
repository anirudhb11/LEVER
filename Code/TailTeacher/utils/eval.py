from xclib.utils.sparse import csr_from_arrays
from xclib.data import data_utils
import xclib.evaluation.xc_metrics as xc_metrics
from xclib.utils.shortlist import Shortlist
from xclib.utils.matrix import SMatrix
from contextlib import contextmanager
from sklearn.preprocessing import normalize
from tqdm import trange, tqdm
import torch
import numpy as np
import wandb
import scipy.sparse as sp
import os
import gc

@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()

def get_filter_map(fname):
    """Load filter file as numpy array"""
    if fname is not None and fname != "":
        return np.loadtxt(fname).astype(np.int32)
    else:
        return None


def filter_predictions(pred, mapping):
    """Filter predictions using given mapping"""
    if mapping is not None and len(mapping) > 0:
        print("Filtering labels.")
        pred[mapping[:, 0], mapping[:, 1]] = 0
        pred.eliminate_zeros()
    return pred


# +
def prepare_metrics_dict(acc, recall, prefix):
    '''
    4 * 5 numpy array [P, nDCG, PSP, PSnDCG]
    '''
    metrics = ['P', 'nDCG', 'PSP', 'PSnDCG']
    metrics_dict = {}
    ks  = [1, 3, 5, 10, 50, 100]
    recall_k = [1 , 3, 5, 10, 20, 30, 50, 75, 100]
#     print(acc.shape)
    print(len(acc), len(acc[0]))
    for metrics_indx in range(4):
        for k in ks:
#             print(f'{prefix}_{metrics[metrics_indx]}@{k}')
            metrics_dict[f'{prefix}_{metrics[metrics_indx]}@{k}'] = acc[metrics_indx][k -1]
    for k in recall_k:
        metrics_dict[f'{prefix}_R@{k}'] = recall[k - 1]
    return metrics_dict
    
def evaluate(_true, _pred, _train, k, A, B, recall_only=False, evaluation_on='tst'):
    """Evaluate function
    * use recall_only = True when dataset is large
    * k: used only for recall@k (precision etc., are computed till @5)
    """
    _true.indices = _true.indices.astype('int64')
    if not recall_only:
        inv_propen = xc_metrics.compute_inv_propesity(_train, A, B)
        acc = xc_metrics.Metrics(_true, inv_propen)
        acc = acc.eval(_pred, 100)
    else:
        print("Only R@k is computed. Don't be surprised with 0 val of others")
        acc = np.zeros((4, 5))
    
    p = xc_metrics.format(*acc)
    rec = xc_metrics.recall(_pred, _true, k)*100  # get the recall
    metrics_dict = prepare_metrics_dict(acc, rec, evaluation_on)
    return f"{p}\nR@1: {rec[0]}\nR@3: {rec[2]}\nR@5: {rec[4]}\nR@10: {rec[9]}\nR@20: {rec[19]}\nR@100: {rec[99]}\n", metrics_dict

def evaluate_with_filter(true_labels, predicted_labels,
                         train_labels, filter_labels, k,
                         A, B, recall_only, evaluation_type):
    """Evaluate function with support of filter file
    * use recall_only = True when dataset is large
    * k: used only for recall@k (precision etc., are computed till @5)
    """
    mapping = get_filter_map(filter_labels)
    predicted_labels = filter_predictions(predicted_labels, mapping)
    return evaluate(
        true_labels, predicted_labels, train_labels, k, A, B, recall_only, evaluation_type)


def _predict_anns(X, clf, k, M=100, efC=300):
    """
    Train a nearest neighbor structure on label embeddings
    - for a given test point: query the graph for closest label
    - HNSW graph would return cosine distance between and document and
    """
    num_instances, num_labels = len(X), len(clf)
    graph = Shortlist(
        method='hnswlib', M=M, efC=efC, efS=k,
        num_neighbours=k, space='cosine', num_threads=64)    
    print("Training ANNS")
    graph.fit(clf)
    print("Predicting using ANNS")
    ind, sim = graph.query(X)
    pred = csr_from_arrays(ind, sim, (num_instances, num_labels))
    return pred


def _predict_ova(X, clf, k=20, batch_size=100, device="cuda", return_sparse=True):
    """Predictions in brute-force manner"""
    torch.set_grad_enabled(False)
    num_instances, num_labels = len(X), len(clf)
    batches = np.array_split(range(num_instances), num_instances//batch_size)
    output = SMatrix(
        n_rows=num_instances,
        n_cols=num_labels,
        nnz=k)
    X = torch.from_numpy(X)        
    clf = torch.from_numpy(clf).to(device).T   
    for ind in tqdm(batches):
        s_ind, e_ind = ind[0], ind[-1] + 1
        _X = X[s_ind: e_ind].to(device)
        ans = _X @ clf
        vals, ind = torch.topk(
            ans, k=k, dim=-1, sorted=True)
        output.update_block(
            s_ind, ind.cpu().numpy(), vals.cpu().numpy())
        del _X
    if return_sparse:
        return output.data()
    else:
        return output.data('dense')[0]


def predict_and_eval(features, clf, labels,
                     trn_labels, filter_labels,
                     A, B, k=10, mode='ova', huge=False, normalize_repr=True, evaluation_type=None):
    """
    Predict on validation set and evaluate
    * support for filter file (pass "" or empty file otherwise)
    * ova will get top-k predictions but anns would get 300 (change if required)"""
    mode='anns' if huge else mode
    if mode == 'ova':
        if normalize_repr:
            
            pred = _predict_ova(normalize(features, copy=True), normalize(clf, copy=True), k=k)
        else:
            print("Predicting w/o normalize")
            pred = _predict_ova(features.copy(), clf.copy(), k=k)
    else:
        pred = _predict_anns(features, clf, k=300)
    labels.indices = labels.indices.astype('int64')
    res = evaluate_with_filter(labels, pred, trn_labels, filter_labels, k, A, B, huge, evaluation_type)
    print(res[0])
    return res[1], pred

def get_embeddings(tokenization_folder, prefix, num_Z, model, max_len, bsz=2000):
    """Get embeddings for given tokenized files"""
    input_ids = np.memmap(
        f"{tokenization_folder}/{prefix}_input_ids.dat",
        mode='r', shape=(num_Z, max_len), dtype=np.int64)
    attention_mask = np.memmap(
        f"{tokenization_folder}/{prefix}_attention_mask.dat",
        mode='r', shape=(num_Z, max_len), dtype=np.int64)
    with evaluating(model), torch.no_grad():
        for i in trange(0, num_Z, bsz):
            batch_input_ids = torch.LongTensor(input_ids[i: i + bsz])
            batch_attention_mask = torch.LongTensor(attention_mask[i: i + bsz])
            _batch_embeddings = model(
                batch_input_ids, batch_attention_mask, None, None).cpu().numpy()
            if(i == 0):
                embeddings = np.zeros((num_Z, _batch_embeddings.shape[1]))
            embeddings[i: i + batch_input_ids.shape[0]] = _batch_embeddings
    return embeddings


def validate(args, snet, trn_X_Y, mode='ova', epoch = None):
    val_X_Y = data_utils.read_gen_sparse(
        os.path.join(args.data_dir, args.val_lbl_fname))
    label_embeddings = get_embeddings(
        args.tokenization_folder, "lbl",
        val_X_Y.shape[1],
        snet,
        args.max_length)
    val_doc_embeddings = get_embeddings(
        args.tokenization_folder,
        f"{args.val_prefix}_doc",
        val_X_Y.shape[0],
        snet,
        args.max_length)
    if not args.huge:
        # train embeddings are computed only when huge is false
        trn_doc_embeddings = get_embeddings(
            args.tokenization_folder, "trn_doc",
            trn_X_Y.shape[0], snet, args.max_length)
        np.save(
            os.path.join(args.result_dir, 'embeddings', f'trn_epoch_{epoch}.ngame.npy'),
            trn_doc_embeddings)
    
    np.save(
        os.path.join(
            args.result_dir, 'embeddings', f'{args.val_prefix}_epoch_{epoch}.ngame.npy'),
        val_doc_embeddings)
    np.save(
        os.path.join(args.result_dir, 'embeddings', f'lbl_epoch_{epoch}.ngame.npy'),
        label_embeddings)
    if args.filter_labels == "":
        filter_labels = None
    else:
        filter_labels = os.path.join(args.data_dir, args.filter_labels)
    res, pred = predict_and_eval(
        val_doc_embeddings, label_embeddings,
        val_X_Y, trn_X_Y, filter_labels,
        A=args.A, B=args.B,
        k=args.k, mode=mode, huge=args.huge, normalize_repr =True, evaluation_type='tst')
    trn_res, trn_pred = predict_and_eval(
        trn_doc_embeddings, label_embeddings,
        trn_X_Y, trn_X_Y, None,
        A=args.A, B=args.B,
        k=args.k, mode=mode, huge=args.huge, normalize_repr=True, evaluation_type='trn'
    )
    sp.save_npz(os.path.join(args.result_dir, f'preds_epoch_{epoch}.npz'), pred)
    sp.save_npz(os.path.join(args.result_dir, f'trn_preds_epoch_{epoch}.npz'), trn_pred)
   
    del val_doc_embeddings, label_embeddings, trn_doc_embeddings
    gc.collect()
    return res, pred, trn_res

def log_into_wandb(res, trn_res, scaler, mean_epoch_loss, mean_epoch_viols, epoch, optimizer):
    # violators, loss, trn metrics, tst metrics, epoch, scaler
    wandb_dict = {}
    wandb_dict.update(trn_res)
    wandb_dict.update(res)
    wandb_dict.update({
        'scalar': scaler,
        'mean_loss': mean_epoch_loss,
        'mean_viol': mean_epoch_viols, 
        'epoch': epoch,
        'lr': optimizer.param_groups[0]['lr']
    })
    wandb.log(wandb_dict)

def dump_in_file(results_dir, content):
    with open(f'{results_dir}/info.log', 'a') as f:
        f.write(content)
        f.write('\n')



