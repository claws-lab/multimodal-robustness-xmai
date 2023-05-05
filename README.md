# XMAI for Multimodal Robustness
### Repository for ACL'23 Paper: "Cross-Modal Attribute Insertions for Assessing the Robustness of Vision-and-Language Learning"
We provide an easy to follow repository with guided notebooks detailing our baselines, method, and evaluation.

## Datasets and Preprocessed Data
The dataset subsets can be downloaded here:
- [MSCOCO Validation 2017](https://cocodataset.org/#download)
- [SNLI-VE Test](https://github.com/OFA-Sys/OFA/blob/main/datasets.md)

To allow for rapid experimentation we provide pre-computed objects and attributes for each dataset:
- MSCOCO Validation 2017: [repo](processed_data/coco_objs_attrs_captions.csv)
- SNLI-VE Test: Large File (will be shared after anonymity period) ***Necessary for evaluation***

## Recommended Organization
```bash
XMAI
├───colab_notebooks
│   ├───augmentation
│   │       TextAttack_baselines.ipynb
│   │       XMAI.ipynb
│   │
│   └───evaluation
│           CLIP_MSCOCO.ipynb
│           METER_SNLI_VE.ipynb
│
├───images
│       coco_val2017.zip
│
├───modified_captions
│   ├───MSCOCO
│   │   ├───baselines
│   │   │       coco_check_modified.csv
│   │   │       coco_clare_modified.csv
│   │   │       coco_del_modified.csv
│   │   │       coco_eda_modified.csv
│   │   │
│   │   └───XMAI
│   └───SNLI_VE
│       ├───baselines
│       │       check_modified_hypothesis.csv
│       │       clare_modified_hypothesis.csv
│       │       del_modified_hypothesis.csv
│       │       eda_modified_hypothesis.csv
│       │
│       └───XMAI
├───object_detection
│       augment_caption.ipynb
│
└───processed_data
        coco_objs_attrs_captions.csv
        snli_ve_test_objects_attrs.csv
```


## Object and Attribute Detection
To perform object and attribute detection yourself:
1. Setup [Bottom-Up Attention Repo](https://github.com/MILVLG/bottom-up-attention.pytorch)
2. Download pretrained model, we use a caffe version from the above repo (may need to modify model loading in notebook depending on which one is used)
3. Run `augment_caption.ipynb` within repo

## Augmentation
### Baselines
- The baseline notebook can be found here: [`colab_notebooks/augmentation/TextAttack_baselines.ipynb`](colab_notebooks/augmentation/TextAttack_baselines.ipynb)
- We provide our resulting files under: [`modified_captions/`](modified_captions/)

### XMAI Method
- The XMAI notebook can be found here: [`colab_notebooks/augmentation/TextAttack_baselines.ipynb`](colab_notebooks/augmentation/TextAttack_baselines.ipynb)
- We provide our main experiment results under:  [`modified_captions/`](modified_captions/)

### Evaluation
- We provide seperate evaluation scripts for MSCOCO and SNLI-VE datasets in [`colab_notebooks/evaluation/`](colab_notebooks/evaluation/)

## Acknowledgements
We also thank the authors and contributors of the following repositories:

```
@misc{yu2020buapt,
  author = {Yu, Zhou and Li, Jing and Luo, Tongan and Yu, Jun},
  title = {A PyTorch Implementation of Bottom-Up-Attention},
  howpublished = {\url{https://github.com/MILVLG/bottom-up-attention.pytorch}},
  year = {2020}
}

@inproceedings{dou2022meter,
  title={An Empirical Study of Training End-to-End Vision-and-Language Transformers},
  author={Dou, Zi-Yi and Xu, Yichong and Gan, Zhe and Wang, Jianfeng and Wang, Shuohang and Wang, Lijuan and Zhu, Chenguang and Zhang, Pengchuan and Yuan, Lu and Peng, Nanyun and Liu, Zicheng and Zeng, Michael},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022},
  url={https://arxiv.org/abs/2111.02387},
}

@article{wang2022ofa,
  author    = {Peng Wang and
               An Yang and
               Rui Men and
               Junyang Lin and
               Shuai Bai and
               Zhikang Li and
               Jianxin Ma and
               Chang Zhou and
               Jingren Zhou and
               Hongxia Yang},
  title     = {OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence
               Learning Framework},
  journal   = {CoRR},
  volume    = {abs/2202.03052},
  year      = {2022}
}
```
