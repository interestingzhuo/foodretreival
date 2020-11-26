##
# foodretreival
Ingradient guided graph for food image retrieval

---
## What can I find here?

This repository contains all code and implementations used in:

```
Ingradient guided graph for food image retrieval(have not submit)
```



**Link**: https://arxiv.org



**Contact**: Karsten Roth, karsten.rh1@gmail.com  

*Suggestions are always welcome!*

---
## Some Notes:

If you use this code in your research, please cite
```

}
```

---

**[All implemented methods and metrics are listed at the bottom!](#-implemented-methods)**

---

## Paper-related Information


---

## How to use this Repo

### Requirements:

* PyTorch 1.2.0+ & Faiss-Gpu
* Python 3.6+
* pretrainedmodels, torchvision 0.3.0+
* numpy, PIL, oprncv-python

An exemplary setup of a virtual environment containing everything needed:
```
(1) wget  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
(2) bash Miniconda3-latest-Linux-x86_64.sh (say yes to append path to bashrc)
(3) source .bashrc
(4) conda create -n DL python=3.6
(5) conda activate DL
(6) conda install matplotlib scipy scikit-learn scikit-image tqdm pandas pillow
(7) conda install pytorch torchvision faiss-gpu cudatoolkit=10.0 -c pytorch
(8) pip install wandb pretrainedmodels
(9) Run the scripts!
```

### Datasets:
Data for
* CUB200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200.html)
* CARS196 (https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* Stanford Online Products (http://cvgl.stanford.edu/projects/lifted_struct/)

can be downloaded either from the respective project sites or directly via Dropbox:

* CUB200-2011 (1.08 GB): https://www.dropbox.com/s/tjhf7fbxw5f9u0q/cub200.tar?dl=0
* CARS196 (1.86 GB): https://www.dropbox.com/s/zi2o92hzqekbmef/cars196.tar?dl=0
* SOP (2.84 GB): https://www.dropbox.com/s/fu8dgxulf10hns9/online_products.tar?dl=0

**The latter ensures that the folder structure is already consistent with this pipeline and the dataloaders**.   

Otherwise, please make sure that the datasets have the following internal structure:

* For CUB200-2011/CARS196:
```
cub200/cars196
└───images
|    └───001.Black_footed_Albatross
|           │   Black_Footed_Albatross_0001_796111
|           │   ...
|    ...
```

* For Stanford Online Products:
```
online_products
└───images
|    └───bicycle_final
|           │   111085122871_0.jpg
|    ...
|
└───Info_Files
|    │   bicycle.txt
|    │   ...
```

Assuming your folder is placed in e.g. `<$datapath/cub200>`, pass `$datapath` as input to `--source`.

### Training:


**[I.]** **A basic sample run using default parameters would like this**:

```
python main.py --loss margin --batch_mining distance --log_online \
              --project DML_Project --group Margin_with_Distance --seed 0 \
              --gpu 0 --bs 112 --data_sampler class_random --samples_per_class 2 \
              --arch resnet50_frozen_normalize --source $datapath --n_epochs 150 \
              --lr 0.00001 --embed_dim 128 --evaluate_on_gpu
```

The purpose of each flag explained:

* `--loss <loss_name>`: Name of the training objective used. See folder `criteria` for implementations of these methods.
* `--batch_mining <batchminer_name>`: Name of the batch-miner to use (for tuple-based ranking methods). See folder `batch_mining` for implementations of these methods.
* `--log_online`: Log metrics online via either W&B (Default) or CometML. Regardless, plots, weights and parameters are all stored offline as well.
*  `--project`, `--group`: Project name as well as name of the run. Different seeds will be logged into the same `--group` online. The group as well as the used seed also define the local savename.
* `--seed`, `--gpu`, `--source`: Basic Parameters setting the training seed, the used GPU and the path to the parent folder containing the respective Datasets.
* `--arch`: The utilized backbone, e.g. ResNet50. You can append `_frozen` and `_normalize` to the name to ensure that BatchNorm layers are frozen and embeddings are normalized, respectively.
* `--data_sampler`, `--samples_per_class`: How to construct a batch. The default method, `class_random`, selects classes at random and places `<samples_per_class>` samples into the batch until the batch is filled.
* `--lr`, `--n_epochs`, `--bs` ,`--embed_dim`: Learning rate, number of training epochs, the batchsize and the embedding dimensionality.  
* `--evaluate_on_gpu`: If set, all metrics are computed using the gpu - requires Faiss-GPU and may need additional GPU memory.



**[II.]** **Advanced Runs**:

```
python main.py --loss margin --batch_mining distance --loss_margin_beta 0.6 --miner_distance_lower_cutoff 0.5 ... (basic parameters)
```

* To use specific parameters that are loss, batchminer or e.g. datasampler-related, simply set the respective flag.
* For structure and ease of use, parameters relating to a specifc loss function/batchminer etc. are marked as e.g. `--loss_<lossname>_<parameter_name>`, see `parameters.py`.
* However, every parameter can be called from every class, as all parameters are stored in a shared namespace that is passed to all methods. This makes it easy to create novel fusion losses and the likes.




# Implemented Methods

For a detailed explanation of everything, please refer to the supplementary of our paper!

### DML criteria

* **Angular** [[Deep Metric Learning with Angular Loss](https://arxiv.org/pdf/1708.01682.pdf)] `--loss angular`
* **ArcFace** [[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf)] `--loss arcface`
* **Contrastive** [[Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)] `--loss contrastive`
* **Generalized Lifted Structure** [[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)] `--loss lifted`
* **Histogram** [[Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf)] `--loss histogram`
* **Marginloss** [[Sampling Matters in Deep Embeddings Learning](https://arxiv.org/abs/1706.07567)] `--loss margin`
* **MultiSimilarity** [[Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](https://arxiv.org/abs/1904.06627)] `--loss multisimilarity`
* **N-Pair** [[Improved Deep Metric Learning with Multi-class N-pair Loss Objective](https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective)] `--loss npair`
* **ProxyNCA** [[No Fuss Distance Metric Learning using Proxies](https://arxiv.org/pdf/1703.07464.pdf)] `--loss proxynca`
* **Quadruplet** [[Beyond triplet loss: a deep quadruplet network for person re-identification](https://arxiv.org/abs/1704.01719)] `--loss quadruplet`
* **Signal-to-Noise Ratio (SNR)** [[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](https://arxiv.org/pdf/1904.02616.pdf)] `--loss snr`
* **SoftTriple** [[SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](https://arxiv.org/abs/1909.05235)] `--loss softtriplet`
* **Normalized Softmax** [[Classification is a Strong Baseline for Deep Metric Learning](https://arxiv.org/abs/1811.12649)] `--loss softmax`
* **Triplet** [[Facenet: A unified embedding for face recognition and clustering](https://arxiv.org/abs/1503.03832)] `--loss triplet`

### DML batchminer

* **Random** [[Facenet: A unified embedding for face recognition and clustering](https://arxiv.org/abs/1503.03832)] `--batch_mining random`
* **Semihard** [[Facenet: A unified embedding for face recognition and clustering](https://arxiv.org/abs/1503.03832)] `--batch_mining semihard`
* **Softhard** [https://github.com/Confusezius/Deep-Metric-Learning-Baselines] `--batch_mining softhard`
* **Distance-based** [[Sampling Matters in Deep Embeddings Learning](https://arxiv.org/abs/1706.07567)] `--batch_mining distance`
* **Rho-Distance** [[Revisiting Training Strategies and Generalization Performance in Deep Metric Learning](https://arxiv.org/abs/2002.08473)] `--batch_mining rho_distance`
* **Parametric** [[PADS: Policy-Adapted Sampling for Visual Similarity Learning](https://arxiv.org/abs/2003.11113)] `--batch_mining parametric`

### Architectures

* **ResNet50** [[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)] e.g. `--arch resnet50_frozen_normalize`.
* **Inception-BN** [[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)] e.g. `--arch bninception_normalize_frozen`.
* **GoogLeNet** (torchvision variant w/ BN) [[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)] e.g. `--arch googlenet`.

### Datasets
* **CUB200-2011** [[Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)]  `--dataset cub200`.
* **CARS196** [[Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)] `--dataset cars196`.
* **Stanford Online Products** [[Deep Metric Learning via Lifted Structured Feature Embedding](https://cvgl.stanford.edu/projects/lifted_struct/)] `--dataset online_products`.



### Evaluation Metrics
**Metrics based on Euclidean Distances**
* **Recall@k**: Include R@1 e.g. with `e_recall@1` into the list of evaluation metrics `--evaluation_metrics`.
* **Normalized Mutual Information (NMI)**: Include with `nmi`.
* **F1**: include with `f1`.
* **mAP (class-averaged)**: Include standard mAP at Recall with `mAP_lim`. You may also include `mAP_1000` for mAP limited to Recall@1000, and `mAP_c` limited to mAP at Recall@Max_Num_Samples_Per_Class. Note that all of these are heavily correlated.

**Metrics based on Cosine Similarities** *(not included by default)*
* **Cosine Recall@k**: Cosine-Similarity variant of Recall@k. Include with `c_recall@k` in `--evaluation_metrics`.
* **Cosine Normalized Mutual Information (NMI)**: Include with `c_nmi`.
* **Cosine F1**: include with `c_f1`.
* **Cosine mAP (class-averaged)**: Include cosine similarity mAP at Recall variants with `c_mAP_lim`. You may also include `c_mAP_1000` for mAP limited to Recall@1000, and `c_mAP_c` limited to mAP at Recall@Max_Num_Samples_Per_Class.

**Embedding Space Metrics**
* **Spectral Variance**: This metric refers to the spectral decay metric used in our ICML paper. Include it with `rho_spectrum@1`. To exclude the `k` largest spectral values for a more robust estimate, simply include `rho_spectrum@k+1`. Adding `rho_spectrum@0` logs the whole singular value distribution, and `rho_spectrum@-1` computes KL(q,p) instead of KL(p,q).
* **Mean Intraclass Distance**: Include the mean intraclass distance via `dists@intra`.
* **Mean Interclass Distance**: Include the mean interlcass distance via `dists@inter`.
* **Ratio Intra- to Interclass Distance**: Include the ratio of distances via `dists@intra_over_inter`.
