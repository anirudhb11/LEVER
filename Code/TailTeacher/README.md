
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
