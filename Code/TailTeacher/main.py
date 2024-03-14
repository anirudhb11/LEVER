import numpy as np
import argparse
import os
import torch
import wandb
import inspect
import sys
import gc
from tqdm import tqdm
from dataset import prepare_data
from loss import prepare_loss
from model import prepare_network
from optimizer import prepare_optimizer_and_schedular
from utils import *

def train(args, snet, criterion, optimizer, scheduler, train_loader):
    import time
    print(f'Criterion is: {criterion}')
    if args.huge:
        # disk shouldn't be slow; Adjust result_dir accordingly
        # Azure blob storage is not an option
        print("Make sure the disk to store embedding bank is fast")
        emb_bank = np.memmap(
            filename=os.path.join(args.result_dir, 'trn.async.dat'),
            shape=(len(train_loader.dataset), snet.repr_dims),
            dtype='float32',
            mode='w+')
    else:
        emb_bank = np.zeros(
            (len(train_loader.dataset), snet.repr_dims), 'float32')
    fp = open(os.path.join(args.result_dir, 'logs.txt'), 'w')
    
    vio_history = []
    loss_history = []
    start_time = time.time()
    val_time = 0
    n_iter = 0
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):
        if epoch in args.curr_steps:
            args.cl_size *=2
            print(f"Changing cluster size to {args.cl_size}")
        cl_start_time = time.time()
        if epoch >= args.cl_start:
            if (epoch - args.cl_start) % args.cl_update == 0:
                print(f'Updating clusters with cluster size {args.cl_size} (using stale embeddings)')
                embs = emb_bank.copy()
                tree_depth = int(np.ceil(np.log(embs.shape[0] / args.cl_size) / np.log(2))) + 1
                print(f"tree depth = {tree_depth}")
                cluster_mat = cluster_items(embs, tree_depth, 16).tocsr()
                del embs
                gc.collect()

            print('Updating train order...')
            cmat = cluster_mat[np.random.permutation(cluster_mat.shape[0])]
            train_loader.batch_sampler.sampler.update_order(cmat.indices)
        else:
            train_loader.batch_sampler.sampler.update_order(
                np.random.permutation(len(train_loader.dataset))) 
        cl_end_time = time.time()
        print(f'TIme to cluster: ', cl_end_time - cl_start_time)
        if epoch %5 == 0:
            if args.save_model:
                snet.eval()
                state_dict = {}
                for k, v in snet.state_dict().items():
                    state_dict[k.replace("module.", "")] = v
                torch.save(state_dict, f"{args.model_dir}/epoch_{epoch}_state_dict.pt")
                with open(f"{args.model_dir}/executed_script.py", "w") as fout:
                    print(inspect.getsource(sys.modules[__name__]), file=fout)
                with open(f"{args.model_dir}/executed_script_args.txt", "w") as fout:
                    print(args, file=fout)
        snet.train()
        torch.set_grad_enabled(True)
        pbar = tqdm(train_loader)
        avg_violators = 0
        epoch_viols = []
        epoch_loss = []
                
        start_epoch_time = time.time()
        for data in pbar:
            snet.zero_grad()
            batch_size = data['batch_size'].item()
            with torch.cuda.amp.autocast():
                ip_embeddings, op_embeddings = snet(
                    data['ip_ind'], data['ip_mask'], data['op_ind'], data['op_mask'])
            emb_bank[data['indices']] = ip_embeddings.detach().cpu().numpy()
            if args.loss_type.__contains__('ohnm'):
                if args.num_violators:
                    with torch.cuda.amp.autocast():
                        scores = ip_embeddings @ op_embeddings.T
                        loss, violators = criterion(scores, data['Y'].to(args.device))
                    avg_violators += float(violators) / float(batch_size)
                    vio_history.append(avg_violators*100)
                    epoch_viols.append(avg_violators)
                    epoch_loss.append(loss.item())
                    pbar.set_description("epoch: {}, loss: {:4e}, viol: {:2e}".format(epoch, loss.item(), violators))
                else:
                    with torch.cuda.amp.autocast():
                        scores = ip_embeddings @ op_embeddings.T
                        loss = criterion(scores.type(torch.float32), data['Y'].to(args.device))
                        epoch_loss.append(loss.item())
                    pbar.set_description(f"epoch: {epoch}, loss: {loss.item()}")
            else:
                raise NotImplementedError("")

            loss_history.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            n_iter += 1
        end_epoch_time = time.time()
        print('Time Taken in epoch: ', end_epoch_time - start_epoch_time)
        mean_epoch_loss = np.mean(np.array(epoch_loss))
        mean_epoch_viols = np.mean(np.array(epoch_viols))
        
        if (args.eval_interval != -1 and ((epoch % args.eval_interval == 0) or (epoch == 0 or epoch == args.epochs - 1))):
            _t = time.time()
            res, _, trn_res = validate(args, snet, train_loader.dataset.labels, mode=args.pred_mode, epoch = epoch)

            log_into_wandb(res, trn_res, scaler.get_scale(), mean_epoch_loss, mean_epoch_viols, epoch, optimizer)
            val_time = val_time + time.time() - _t
            fp.write(f"epoch: {epoch}\n{res}")
    if args.huge:
        emb_bank.flush()
    total_time = time.time() - start_time
    pickle.dump({'vio': vio_history, 'loss': loss_history},
        open(os.path.join(args.result_dir, 'train_history.pkl'), 'wb'))
    fp.write(f"Total time: {total_time} sec.\n")
    fp.write(f"Validation time: {val_time}\n")
    fp.write(f"Train time: {total_time - val_time}\n")
    fp.close()

def setup_wandb(args):
    wandb.init(project="Tail-Teacher", entity="anirudhb", name=args.version)
    wandb.config.update(args)


def main(args):
    args.save_model=True
    if args.save_model:
        print("WILL SAVE MODEL")
    args.device = torch.device(args.device)
    args.model_dir = os.path.join(
        args.work_dir, 'models' , "Tail-Teacher", args.dataset, args.version)
    args.result_dir = os.path.join(
        args.work_dir, 'results', "Tail-Teacher", args.dataset, args.version)

    args.data_dir =  os.path.join(
            args.work_dir, 'Datasets', args.dataset)
    args.tokenization_folder = os.path.join(
        args.data_dir, f"{args.tokenizer_type}-{args.max_length}")
    print(args.tokenization_folder)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, 'embeddings'), exist_ok=True)
    dump_in_file(args.result_dir, str(args))
    setup_wandb(args)
    # get curruculum steps
    if args.curr_steps == "":
        args.curr_steps = set()
    else:
        args.curr_steps = set(map(int, args.curr_steps.split(",")))

    # prepare data, network, and optimizer
    train_loader = prepare_data(args)
    criterion = prepare_loss(args)
    snet = prepare_network(args)
    optimizer, scheduler = prepare_optimizer_and_schedular(args, snet, len(train_loader))
    train(args, snet, criterion, optimizer, scheduler, train_loader)
    
    if args.save_model:
        snet.eval()
        state_dict = {}
        for k, v in snet.state_dict().items():
            state_dict[k.replace("module.", "")] = v
        torch.save(state_dict, f"{args.model_dir}/state_dict.pt")
        with open(f"{args.model_dir}/executed_script.py", "w") as fout:
            print(inspect.getsource(sys.modules[__name__]), file=fout)
        with open(f"{args.model_dir}/executed_script_args.txt", "w") as fout:
            print(args, file=fout)


# CUDA_VISIBLE_DEVICES=0,1 python main.py --work-dir /home/t-abuvanesh/xc/ --dataset LF-AmazonTitles-131K --epochs 300 --batch-size 1600 --margin 0.3 --eval-interval 1 --enc-lr 2e-4 --version lfat-131k-lbl-side --filter-labels tst_filter_labels.txt --num-negatives 10 --num-violators --save-model  --batch-type lbl --loss-type ohnm --cl-size 8 --cl-start 10 --cl-update 5 --curr-steps 25,50,75,100,125,150,200
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, help="Work dir")
    parser.add_argument("--dataset", type=str, help="Dataset name", default='LF-AmazonTitles-131K')
    parser.add_argument("-e", "--epochs", type=int, help="The number of epochs to run for", default=600)
    parser.add_argument("-b", "--batch-size", type=int, help="The batch size", default=4096)
    parser.add_argument("-m", "--margin", type=float, help="Margin below which negative labels are not penalized", default=1.0)
    parser.add_argument("-A", type=float, help="The propensity factor A" , default=0.55)
    parser.add_argument("-B", type=float, help="The propensity factor B", default=1.5)
    parser.add_argument("--device", type=str, help="device to run", default="cuda")
    parser.add_argument("--enc-lr", type=float, help="encoder learning rate", default=0.0002)
    parser.add_argument("--enc-wd", type=float, help='encoder weight decay', default= 0.01)    
    parser.add_argument("--pred-mode", type=str, help="ova or anns", default='ova')
    parser.add_argument("--huge", action='store_true', help="Compute only recall; don't save train; use memmap")
    parser.add_argument("--save-model", action='store_true', required=False, help="Should the model be saved")
    parser.add_argument("--trn-lbl-fname", type=str, required=False, help="Train label file name", default="trn_X_Y.txt")
    parser.add_argument("--val-lbl-fname", type=str, required=False, help="Train label file name", default="tst_X_Y.txt")
    parser.add_argument("--val_prefix", type=str, required=False, help="Train label file name", default="tst")
    parser.add_argument("--file-type", type=str, required=False, help="File type txt/npz", default="txt")
    parser.add_argument("--version", type=str, help="Version of the run", default="0")
    parser.add_argument("--filter-labels", type=str, help="filter labels at validation time", default="tst_filter_labels.txt")
    parser.add_argument("--curr-steps", type=str, help="double cluster size at each step (csv)", default="")
    parser.add_argument("--eval-interval", type=int, help="The numbers of epochs between acc evalulation", default=30)
    parser.add_argument("--num-violators", action="store_true", help="Should average number of violators be printed")
    parser.add_argument("--agressive-loss", action="store_true", help="Should average number of violators be printed")
    parser.add_argument("--loss-type", type=str, help="Squared or sqrt", default='ohnm')
    parser.add_argument("--k", type=int, help="k for recall", default=100)
    parser.add_argument("--batch-type", type=str, help="doc/lbl", default='doc')
    parser.add_argument("--max-length", type=int, help="Max length for tokenizer", default=32)
    parser.add_argument("--num-negatives", type=int, help="Number of negatives to use", default=3)
    parser.add_argument("--tokenizer-type", type=str, help="Tokenizer to use", default="bert-base-uncased")
    parser.add_argument("--encoder-name", type=str, help="Encoder to use", default="msmarco-distilbert-base-v4")
    parser.add_argument("--transform-dim", type=int, help="Transform bert embeddings to size", default=-1)
    parser.add_argument("--cl-size", type=int, help="cluster size", default=32)
    parser.add_argument("--cl-start", type=int, help="", default=999999)
    parser.add_argument("--cl-update", type=int, help="", default=5)
    parser.add_argument("--warmup-steps", type=int, default=100, help='number of steps to warmup for')
    parser.add_argument("--loss-reduction", type=str, default='mean')
    args = parser.parse_args()
    print(args)
    main(args)
