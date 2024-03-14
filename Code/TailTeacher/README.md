## Directory Structure

The code provided assumes the following directory structure.

```txt
+-- <work_dir>
|  +-- Datasets
|  |  +-- <dataset>
|  +-- models
|  +-- results
```

## Data Preparation
You can download the datasets from the [XML repo](http://manikvarma.org/downloads/XC/XMLRepository.html).

A dataset folder should have the following directory structure. Below we show it for LF-AmazonTitles-131K dataset:

```bash
📁 LF-AmazonTitles-131K/
    📄 trn_X_Y.txt # contains mappings from train IDs to label IDs
    📄 trn_filter_labels.txt # this contains train reciprocal pairs to be ignored in evaluation
    📄 tst_X_Y.txt # contains mappings from test IDs to label IDs
    📄 tst_filter_labels.txt # this contains test reciprocal pairs to be ignored in evaluation
    📄 trn_X.txt # each line contains the raw input train text, this needs to be tokenized
    📄 tst_X.txt # each line contains the raw input test text, this needs to be tokenized
    📄 Y.txt # each line contains the raw label text, this needs to be tokenized
```

To tokenize the raw train, test and label texts, we can use the following command (change the path of the dataset folder accordingly):
```bash
python -W ignore -u utils/CreateTokenizedFiles.py \
--data-dir xc/Datasets/LF-AmazonTitles-131K \
--max-length 32 \
--tokenizer-type bert-base-uncased \
--tokenize-label-texts
```

## Training the Tail Robust Teacher

LEVER builds upon NGAME's [1] Module-I code base. Please refer to the sample command below to train the Siamese Teacher model. We use the same hyper-parameters as described in [1] to train the teacher model.

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py --work-dir <work-dir> --dataset LF-AmazonTitles-131K --epochs 300 --batch-size 1600 --margin 0.3 --eval-interval 1 --enc-lr 2e-4 --version lfat-131k-lbl-side --filter-labels tst_filter_labels.txt --num-negatives 10 --num-violators --save-model  --batch-type lbl --loss-type ohnm --cl-size 8 --cl-start 10 --cl-update 5 --curr-steps 25,50,75,100,125,150,200
```

## References

[1]: K. Dahiya, N. Gupta, D. Saini, A. Soni, Y. Wang, K. Dave, J. Jiao, K. Gururaj, P. Dey, A. Singh, D. Hada, V. Jain, B. Paliwal, A. Mittal, S. Mehta, R. Ramjee, S. Agarwal, P. Kar and M. Varma. NGAME: Negative mining-aware mini-batching for extreme classification. In WSDM, Singapore, March 2023.