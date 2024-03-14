import numpy as np
import scipy.sparse as sp
import argparse
import hnswlib
import numba as nb

class HNSW(object):
    def __init__(self, M, efC, efS, dim, num_threads):
        self.index = hnswlib.Index(space="ip", dim=dim)
        self.M = M
        self.num_threads = num_threads
        self.efC = efC
        self.efS = efS

    def fit(self, data, print_progress=True):
        print('Fitting HNSW index')
        self.index.init_index(
            max_elements=data.shape[0], ef_construction=self.efC, M=self.M
        )
        data_labels = np.arange(data.shape[0]).astype(np.int64)
        self.index.add_items(data, data_labels, num_threads=self.num_threads)

    def predict(self, data, num_nbrs=None):
        print('Predicting using HNSW index')
        self.index.set_ef(self.efS)
        if num_nbrs is None:
            num_nbrs = self.efS
        if num_nbrs > self.efS:
            print(f"num_nbrs > efS. efS={self.efS} number of nbrs will be returned")
            num_nbrs = self.efS
        indices, distances = self.index.knn_query(data, k=num_nbrs)
        indices = indices.astype(np.int64)
        return indices, distances

    def save(self, fname):
        self.index.save_index(fname)

    def load(self, fname):
        self.index.load_index(fname)

def get_nbr_indices_and_distances():
    trn_doc_embs = np.load(args.trn_doc_emb_fpth)
    lbl_embs = np.load(args.lbl_emb_fpth)
    index_on_docs = HNSW(M=100, efC=300, efS=500, num_threads=96, dim=768)
    index_on_lbls = HNSW(M=100, efC=300, efS=500, num_threads=96, dim=768)
    index_on_docs.fit(trn_doc_embs)
    index_on_lbls.fit(lbl_embs)
    
    doc_indices, doc_distances = index_on_docs.predict(lbl_embs, num_nbrs=100)
    lbl_indices, lbl_distances = index_on_lbls.predict(lbl_embs, num_nbrs=100)
    return doc_indices, doc_distances, lbl_indices, lbl_distances

@nb.njit(parallel=True)
def augment_docs(true_labels_indices, true_labels_indptr, sim_threshold, ppl_threshold, similarity, indices, sorted_label_index, pts_per_lbl):
    num_labels = similarity.shape[0]
    max_elts=  num_labels * ppl_threshold
    aug_data_pts = np.zeros(max_elts) - 1
    aug_labels =  np.zeros(max_elts) - 1
    aug_vals = np.zeros(max_elts) - 1
    for i in nb.prange(num_labels):
        lbl_index = sorted_label_index[i]
        if pts_per_lbl[lbl_index] >= ppl_threshold: # num_ngbrs
            break
        pts_added_for_label = pts_per_lbl[lbl_index]
        idx = 0
        base = i * ppl_threshold
        while pts_added_for_label < ppl_threshold:
            if idx == len(indices[lbl_index]):
                break
            if similarity[lbl_index][idx] < sim_threshold:
                break
            trn_pt_idx = indices[lbl_index][idx]
            if lbl_index in true_labels_indices[true_labels_indptr[trn_pt_idx] : true_labels_indptr[trn_pt_idx + 1]]:
                idx += 1
                continue
            aug_data_pts[base + idx] = trn_pt_idx
            aug_labels[base + idx] = lbl_index
            aug_vals[base + idx] = similarity[lbl_index][idx]
            idx += 1
            pts_added_for_label += 1

    inds = np.where(aug_data_pts >= 0)
    aug_data_pts = aug_data_pts[inds]
    aug_labels = aug_labels[inds]
    aug_vals = aug_vals[inds]
    return aug_data_pts, aug_labels, aug_vals

@nb.njit(parallel=True)
def augment_lbls(true_labels_indices, true_labels_indptr, sim_threshold, ppl_threshold, similarity, indices, offset):
    num_labels = similarity.shape[0]
    max_elts=  num_labels * ppl_threshold
    aug_data_pts = np.zeros(max_elts) - 1
    aug_labels =  np.zeros(max_elts) - 1
    aug_vals = np.zeros(max_elts) - 1
    for lbl_index in nb.prange(num_labels):
        pts_added_for_label = 1 # one label is already gt here
        idx = 0
        base = lbl_index * ppl_threshold
        while pts_added_for_label < ppl_threshold:
            if idx == len(indices[lbl_index]):
                break

            if similarity[lbl_index][idx] < sim_threshold:
                break

            trn_pt_idx = lbl_index + offset # offset due to already training pts.
            if indices[lbl_index][idx] in true_labels_indices[true_labels_indptr[trn_pt_idx] : true_labels_indptr[trn_pt_idx + 1]]:
                idx += 1
                continue

            aug_data_pts[base + idx] = trn_pt_idx
            aug_labels[base + idx] = indices[lbl_index][idx]
            aug_vals[base + idx] = similarity[lbl_index][idx]

            idx += 1
            pts_added_for_label += 1
    inds = np.where(aug_data_pts >= 0)
    aug_data_pts = aug_data_pts[inds]
    aug_labels = aug_labels[inds]
    aug_vals = aug_vals[inds]
    return aug_data_pts, aug_labels, aug_vals

def main(args):
    doc_indices, doc_distances, lbl_indices, lbl_distances = get_nbr_indices_and_distances()
    trn_X_Y = sp.load_npz(f'{args.work_dir}/Datasets/{args.dataset}/trn_X_Y.npz')
    doc_similarities = (2 - doc_distances) / 2
    lbl_similarities = (2 - lbl_distances) / 2
    if args.score_transformer == 'sigmoid':
        doc_similarities = 1 / (1 + np.exp(- args.sigmoid_a * doc_similarities + args.sigmoid_b))
        lbl_similarities = 1 / (1 + np.exp(- args.sigmoid_a * -lbl_similarities + args.sigmoid_b))
    
    pts_per_lbl = np.array(trn_X_Y.sum(axis=0), dtype=int).flatten()
    sorted_label_index = np.argsort(pts_per_lbl)

    lbl_data = trn_X_Y.tocoo()
    row_idx, col_idx, vals = lbl_data.row, lbl_data.col, lbl_data.data
    doc_aug_data_pts, doc_aug_labels, doc_aug_vals = augment_docs(trn_X_Y.indices.astype(np.int64), trn_X_Y.indptr.astype(np.int64), args.doc_similarity_threshold, args.num_docs, doc_similarities, doc_indices, sorted_label_index.astype(np.int32), pts_per_lbl.astype(np.int32))
    lbl_aug_data_pts, lbl_aug_labels, lbl_aug_vals = augment_lbls(trn_X_Y.indices.astype(np.int64), trn_X_Y.indptr.astype(np.int64), args.lbl_similar_threshold, args.num_lbls, lbl_similarities, lbl_indices, trn_X_Y.shape[0])
    row_idx = np.concatenate([row_idx, doc_aug_data_pts, lbl_aug_data_pts], axis = 0)
    col_idx = np.concatenate([col_idx, doc_aug_labels, lbl_aug_labels], axis= 0)
    vals = np.concatenate([vals, doc_aug_vals, lbl_aug_vals], axis = 0)
    trn_X_Y = sp.csr_matrix((vals, (row_idx, col_idx)), shape=lbl_data.shape)
    sp.save_npz(f'./lfat_131k_lbl_ngame.npz', trn_X_Y)
    
# python teacher_augmented_dataset.py --work-dir <work-dir> --trn-doc-emb-fpth  <trn-doc-emb-path> --lbl-emb-fpth  <lbl-emb-pth> --num-docs 0 --doc-similarity-threshold 0.8 --num-lbls 4 --lbl-similar-threshold 0.8
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='LF-AmazonTitles-131K-Aug', help='Dataset name, use the L-L Augmented dataset')
    parser.add_argument('--trn-doc-emb-fpth', type=str, default='trn_doc_embeddings.npy', help='Training document embeddings file path')
    parser.add_argument('--lbl-emb-fpth', type=str, default='lbl_embeddings.npy', help='Label embeddings file path')
    parser.add_argument('--num-docs', type=int, help='Number of training documents that should be added for each label')
    parser.add_argument('--doc-similarity-threshold', type=float, default=0.8, help='Threshold for document similarity')
    parser.add_argument('--num-lbls', type=int, help='Number of labels to be added for each label')
    parser.add_argument('--lbl-similar-threshold', type=float, default=0.8, help='Threshold for label similarity')
    parser.add_argument('--score-transformer', choices=['sigmoid', 'linear'], type=str, default='linear', help='Score transformation function to use on cosine similarity')
    parser.add_argument('--sigmoid-a', type=float, default=1, help='scale parameter for sigmoid transformation')
    parser.add_argument('--sigmoid-b', type=float, default=0, help='shift parameter for sigmoid transformation')
    args = parser.parse_args()
    main(args)