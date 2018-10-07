# Human Protein Atlas Image Classification Challenge

[Competition](https://www.kaggle.com/c/human-protein-atlas-image-classification)

## Experiments

### Simple Convnet

Baseline convolutional approach to get familiar with the data.

Date|Epochs Trained|Augmentation|Hyperparameters|Val Macro F1|Test Macro F1|
|---|---|---|---|---|---|
2018-10-06|TBD|No|[link](hyperparameters/simple_convnet_v1.json)|TBD|TBD|

### Triplet Model

Due to the extreme class imbalance, it could make sense to treat this as a few-shot learning task. Train an embedding using Triplet loss, and query the training set using something like nearest neighbors to classify new samples.

Date|Epochs Trained|Augmentation|Hyperparameters|Val Macro F1|Test Macro F1|
|---|---|---|---|---|---|