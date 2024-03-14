## Creating the Augmented Dataset

Once the Tail Robust Teacher has been trained we prepare a dataset that encodes the knowledge of the tail robust teacher. To create the dataset do following steps:

* Run [`create_aug_dataset.ipynb`](./create_aug_dataset.ipynb) to create a dataset that has adds each label as a data point and it's relevant label as itself.

* Run [`create_teacher_distilled_dataset.py](./create_teacher_distilled_dataset.py) to create a dataset that uses the high-scoring data point label pairs (as per the Siamese Teacher) to augment the ground truth.


## Training an Extreme Classifier

Using the new trn_X_Y train any extreme classification method.