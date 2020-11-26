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



**Contact**:  

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

* PyTorch 1.2.0+ 
* Python 3.6+
* torchvision 0.3.0+
* numpy, PIL, oprncv-python



### Datasets:
Data for
* food101 ()
* food172 ()
* food200 ()


### Training:

The purpose of each flag explained:

* `--loss <loss_name>`: Name of the training objective used. See folder `criteria` for implementations of these methods.



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

* **smoothap** [[Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval](https://arxiv.org/abs/2007.12163)] `--loss smoothap`
* **Circle loss** [] `--loss circle`
* **Contrastive** [] `--loss contrastive`
* **Triplet** [] `--loss triplet`


### Architectures

* **ResNet50&101** [[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)] e.g. `--arch resnet50_frozen_normalize`.
