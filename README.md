# Human Protein Atlas Image Classification Challenge

[Competition](https://www.kaggle.com/c/human-protein-atlas-image-classification)

## Experiments

### Simple Convnet

Baseline convolutional approach to get familiar with the data. Uses green channel only.

Date|Commit|Epochs Trained|Augmentation|Artifacts|Val Macro F1|Test Macro F1|
|---|---|---|---|---|---|---|
2018-10-06|[1a99729](https://github.com/bharris47/human-protein-atlas/commit/1a99729dab0a660003fdf353e80dae4ed5f183c9)|17|No|[link](models/simple_convnet_v1/2018-10-06)|0.363|0.206|
2018-10-07|[172ebf0](https://github.com/bharris47/human-protein-atlas/commit/172ebf028ba1b976158cdc6e84eb5fda0871988a)|23|Yes|[link](models/simple_convnet_v1/2018-10-07-08-48-33)|0.315|0.184|


### Triplet Model

Due to the extreme class imbalance, it could make sense to treat this as a few-shot learning task. Train an embedding using Triplet loss, and query the training set using something like nearest neighbors to classify new samples.
