import torch
import numpy as np
import os
from xclib.data import data_utils
from functools import partial


class MySampler(torch.utils.data.Sampler[int]):
    def __init__(self, order):
        self.order = order.copy()

    def update_order(self, x):
        self.order[:] = x[:]

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)

def _collate_fn(batch):
    batch_labels = []
    random_pos_indices = []
    for item in batch:
        batch_labels.append(item[2])
        random_pos_indices.append(item[3])

    batch_size = len(batch_labels)

    ip_ind = np.vstack([x[0] for x in batch]) 
    ip_mask = np.vstack([x[1] for x in batch])
    op_ind = np.vstack([x[4] for x in batch])
    op_mask = np.vstack([x[5] for x in batch])
    batch_selection = np.zeros((batch_size, batch_size), dtype=np.float32)     

    random_pos_indices_set = set(random_pos_indices)
    random_pos_indices = np.array(random_pos_indices, dtype=np.int32)
    

    for (i, item) in enumerate(batch_labels):
        intersection = set(item).intersection(random_pos_indices_set)
        result = np.zeros(batch_size, dtype=np.float32)
        for idx in intersection:
            result += (idx == random_pos_indices)   
        batch_selection[i] = result 
    return ip_ind, ip_mask, op_ind, op_mask, batch_selection 

def clip_batch_lengths(ind, mask, max_len):
    _max = min(np.max(np.sum(mask, axis=1)), max_len)
    return ind[:, :_max], mask[:, :_max]


def collate_fn(batch, max_len):
    """
    collate function: should support both doc-side and
    label-side training
    * will clip after max_len (change if required)
    """
    batch_data = {}
    batch_size = len(batch)
    batch_data['batch_size'] = torch.tensor(batch_size, dtype=torch.int32)

    ip_ind, ip_mask, op_ind, op_mask, batch_selection = _collate_fn(batch)
    ip_ind, ip_mask = clip_batch_lengths(ip_ind, ip_mask, max_len)
    op_ind, op_mask = clip_batch_lengths(op_ind, op_mask, max_len)

    batch_data['indices'] = torch.LongTensor([item[6] for item in batch])
    batch_data['ip_ind'] = torch.from_numpy(ip_ind)
    batch_data['ip_mask'] = torch.from_numpy(ip_mask)
    batch_data['op_ind'] = torch.from_numpy(op_ind)
    batch_data['op_mask'] = torch.from_numpy(op_mask)
    batch_data['Y'] = torch.from_numpy(batch_selection)
    batch_data['Y_mask'] = None
    return batch_data


class DatasetL(torch.utils.data.Dataset):
    """
    Dataset for label-side training
    """
    def __init__(self, lbl_fname, tokenization_folder, max_len):
        self.max_len = max_len
        self.labels = data_utils.read_gen_sparse(lbl_fname)        
        self.valid_labels = np.where(np.ravel(np.sum(self.labels.astype(np.bool_), axis=0)) > 0)[0]
        print("#valid labels is: {}".format(len(self.valid_labels))) 
        
        self.X_input_ids = np.memmap(
            f"{tokenization_folder}/trn_doc_input_ids.dat",
            mode='r',
            shape=(self.labels.shape[0], max_len),
            dtype=np.int64)
        self.X_attention_mask = np.memmap(
            f"{tokenization_folder}/trn_doc_attention_mask.dat",
            mode='r',
            shape=(self.labels.shape[0], max_len),
            dtype=np.int64)
        
        self.Y_input_ids = np.memmap(
            f"{tokenization_folder}/lbl_input_ids.dat",
            mode='r',
            shape=(self.labels.shape[1], max_len),
            dtype=np.int64)
        self.Y_attention_mask = np.memmap(
            f"{tokenization_folder}/lbl_attention_mask.dat",
            mode='r',
            shape=(self.labels.shape[1], max_len),
            dtype=np.int64)
        self._labels = self.labels.T.tocsr()
    
    def __getitem__(self, index):
        """Get a label at index"""
        # Get a randomly sampled positive data point
        _index = self.valid_labels[index]
        pos_indices = self._labels[_index].indices
        pos_ind = np.random.choice(pos_indices)

        return (self.Y_input_ids[_index], self.Y_attention_mask[_index], 
                pos_indices, pos_ind, self.X_input_ids[pos_ind],
                self.X_attention_mask[pos_ind], index)
   
    def __len__(self):
        return len(self.valid_labels)


class DatasetD(torch.utils.data.Dataset):
    """
    Dataset for document-side training
    """
    def __init__(self, lbl_fname, tokenization_folder, max_len):
        self.max_len = max_len
        self.labels = data_utils.read_gen_sparse(lbl_fname)        
        self.X_input_ids = np.memmap(
            f"{tokenization_folder}/trn_doc_input_ids.dat",
            mode='r',
            shape=(self.labels.shape[0], max_len),
            dtype=np.int64)
        self.X_attention_mask = np.memmap(
            f"{tokenization_folder}/trn_doc_attention_mask.dat",
            mode='r',
            shape=(self.labels.shape[0], max_len),
            dtype=np.int64)
        
        self.Y_input_ids = np.memmap(
            f"{tokenization_folder}/lbl_input_ids.dat",
            mode='r',
            shape=(self.labels.shape[1], max_len),
            dtype=np.int64)
        self.Y_attention_mask = np.memmap(
            f"{tokenization_folder}/lbl_attention_mask.dat",
            mode='r',
            shape=(self.labels.shape[1], max_len),
            dtype=np.int64)

    
    def __getitem__(self, index):
        """Get a label at index"""
        # Get a randomly sampled positive data point
        pos_indices = self.labels[index].indices
        if len(pos_indices) == 0:
            pos_ind = self.Y_input_ids.shape[0] - 1
        else:
            pos_ind = np.random.choice(pos_indices)

        return (self.X_input_ids[index], self.X_attention_mask[index], 
                pos_indices, pos_ind, self.Y_input_ids[pos_ind],
                self.Y_attention_mask[pos_ind], index)
   
    def __len__(self):
        return len(self.X_input_ids)
    
def prepare_data(args):
    if(not(os.path.exists(args.tokenization_folder))):
        print("Please create tokenization memmaps for this "\
        "dataset using CreateTokenizedFiles.py as a one time effort")
        sys.exit(0)

    print("==> Creating Dataloader...")
    if args.batch_type == 'doc':
        train_dataset = DatasetD(
            os.path.join(args.data_dir, args.trn_lbl_fname),
            args.tokenization_folder,
            args.max_length,
            )
    elif args.batch_type == 'lbl':
        train_dataset = DatasetL(
            os.path.join(args.data_dir, args.trn_lbl_fname),
            args.tokenization_folder,
            args.max_length
            )
    else:
        raise NotImplementedError("")

    train_order = np.random.permutation(len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=12,
        prefetch_factor=5,
        collate_fn=partial(collate_fn, max_len=args.max_length),
        batch_sampler=torch.utils.data.sampler.BatchSampler(
            MySampler(train_order), args.batch_size, False))
    return train_loader

if __name__ == '__main__':
    import sys
    datasetL = DatasetL(sys.argv[1], sys.argv[2], 32)