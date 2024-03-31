
### Introduction

Code for our paper "Meta Pairwise Relationship Distillation for Unsupervised Person Re-identification" (improved version).

### Prerequisites

- Necessary packages listed in [requirements.txt](requirements.txt)
- Training Data
  
  (Market-1501, DukeMTMC-reID and MSMT-17. You can download these datasets from [Zhong's repo](https://github.com/zhunzhong07/ECN))

   Unzip all datasets and ensure the file structure is as follow:
   
   ```
   MetaPRD/examples/data    
   │
   └───market1501 OR dukemtmc OR msmt17
        │   
        └───DukeMTMC-reID OR Market-1501-v15.09.15 OR MSMT17_V1
            │   
            └───bounding_box_train
            │   
            └───bounding_box_test
            | 
            └───query
   ```

### Usage

# on Market-1501 (w/ GCN)
python examples/train_mprd.py -b 64 -a resnet50 -d market1501 --iters 400 --momentum 0.1 --eps 0.4 --num-instances 16 --use-hard --use-gcn

# on DukeMTMC-reID (w/ GCN)
python examples/train_mprd.py -b 64 -a resnet50 -d dukemtmcreid --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --use-hard --use-gcn

# on MSMT17 (w/ GCN)
python examples/train_mprd.py -b 64 -a resnet50 -d MSMT17 --iters 400 --momentum 0.1 --eps 0.7 --num-instances 16 --use-hard --use-gcn

If do not want to use the GCN to guide CNN training, you can optimize the CNN by follows
# on Market-1501 (w/o GCN)
python examples/train_mprd.py -b 64 -a resnet50 -d market1501 --iters 400 --momentum 0.1 --eps 0.4 --num-instances 16 --use-hard

# on DukeMTMC-reID (w/o GCN)
python examples/train_mprd.py -b 64 -a resnet50 -d dukemtmcreid --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --use-hard

# on MSMT17 (w/o GCN)
python examples/train_mprd.py -b 64 -a resnet50 -d MSMT17 --iters 400 --momentum 0.1 --eps 0.7 --num-instances 16 --use-hard

### pre-trained model
# for CNN
When training with the backbone of [IBN-ResNet](https://arxiv.org/abs/1807.09441), you need to download the ImageNet-pretrained model from this [link](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) 
The pre-trained model of CNN are saved in examples/pretrained
ImageNet-pretrained models for **ResNet-50** will be automatically downloaded in the python script.
use `-a resnet50` (default) for the backbone of ResNet-50, and `-a resnet_ibn50a` for the backbone of IBN-ResNet.

# for GCN
If you want to use GCN to guide CNN training, we suggest utilizing the pre-trained model for GCN in mprd/gcn/saved_model/pretrained, more details in [there](mprd/gcn/saved_model/pretrained/readme.md).

# Citation

If this repo is helpful for your research, please consider citing the paper:

```BibTeX
@inproceedings{ji2021meta,
  title={Meta Pairwise Relationship Distillation for Unsupervised Person Re-identification},
  author={Ji, Haoxuanye and Wang, Le and Zhou, Sanping and Tang, Wei and Zheng, Nanning and Hua, Gang},
  booktitle={ICCV},
  pages={3661--3670},
  year={2021}
}
```
or
```BibTeX
@article{ji2022meta,
  title={Meta Pairwise Relationship Distillation for Unsupervised Person Re-identification},
  author={Ji, Haoxuanye and Wang, Le and Zhou, Sanping and Tang, Wei and Zheng, Nanning and Hua, Gang},
  booktitle={submitted to T-NNLS},
  year={2022}
}
```

### Acknowledgments
This repo borrows partially from 
[Link-Prediction-Based-on-Graph-Neural-Networks](https://github.com/engineerjkk/Link-Prediction-Based-on-Graph-Neural-Networks),
[SpCL](https://github.com/yxgeee/SpCL) and
[cluster-contrast-reid](https://github.com/alibaba/cluster-contrast-reid). 
If you find our code useful, please cite their papers.

```
@inproceedings{nips2018link,
  title={Link prediction based on graph neural networks},
  author={Zhang, Muhan and Chen, Yixin},
  booktitle={NeurIPS},
  pages={5171--5181},
  year={2018}
}
```

```
@inproceedings{ge2020selfpaced,
    title={Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID},
    author={Yixiao Ge and Feng Zhu and Dapeng Chen and Rui Zhao and Hongsheng Li},
    booktitle={NeurIPS},
    year={2020}
}
```

```
@inproceedings{arxiv2021Cluster,
    author = {Dai, Zuozhuo and Wang, Guangyuan and Yuan, Weihao and Zhu, Siyu and Tan, Ping},
    title = {Cluster Contrast for Unsupervised Person Re-Identification},
    booktitle = {arXiv:2103.11568},
    year = 2021
}
```
