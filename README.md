# XMAI for Multimodal Robustness
### Repository for ACL'23 Paper: "Cross-Modal Attribute Insertions for Assessing the Robustness of Vision-and-Language Learning"

**Authors**: [Shivaen Ramshetty](https://shivaen.org/)\*, [Gaurav Verma](https://gaurav22verma.github.io/)\*, and [Srijan Kumar](https://faculty.cc.gatech.edu/~srijan/)  
**Affiliation**: Georgia Institute of Technology  

**Paper (pdf)**: [arXiv](https://arxiv.org/abs/2306.11065)  
**Poster (pdf)**: [coming soon]()  

# Overview Figure

<p align="center">
    <img src="https://github.com/claws-lab/multimodal-robustness-xmai/blob/main/assets/overview.png" width="40%;">
</p>
    
# Qualitative Examples
<p align="center">
    <img src="https://github.com/claws-lab/multimodal-robustness-xmai/blob/main/assets/qualitative_examples.png" width="80%;">
</p>

# Code, Data, and Resources

We provide an easy to follow repository with guided notebooks detailing our baselines, method, and evaluation.

## Datasets and Preprocessed Data
The dataset subsets can be downloaded here:
- [MSCOCO Validation 2017](https://cocodataset.org/#download)
- [SNLI-VE Test](https://github.com/OFA-Sys/OFA/blob/main/datasets.md)

To allow for rapid experimentation we provide pre-computed objects and attributes for each dataset:
- MSCOCO Validation 2017: [repo](processed_data/coco_objs_attrs_captions.csv)
- SNLI-VE Test: [gdrive](https://drive.google.com/file/d/1iBdsHi3UKIwKMyxV23gvgRnPS33FPwoa/view?usp=sharing)

## Object and Attribute Detection
To perform object and attribute detection yourself:
1. Setup [Bottom-Up Attention Repo](https://github.com/MILVLG/bottom-up-attention.pytorch) or use our [docker](detector)
2. Download pretrained model if setting up yourself.
3. Follow instructions in `detector/README.md` to capture objects and attributes for the above data or your own.

## Augmentation
To augment and evaluate your own data, we provide scripts in [XMAI](XMAI/)

Notebooks and data for our paper are found within [paper_experiments](paper_experiments/)

### Baselines
- The baseline notebook can be found here: [`paper_experiments/colab_notebooks/augmentation/TextAttack_baselines.ipynb`](paper_experiments/colab_notebooks/augmentation/TextAttack_baselines.ipynb)
- We provide our resulting files under: [`paper_experiments/modified_captions/`](paper_experiments/modified_captions/)

### XMAI Method
- The XMAI notebook can be found here: [`paper_experiments/colab_notebooks/augmentation/TextAttack_baselines.ipynb`](paper_experiments/paper_experiments/colab_notebooks/augmentation/TextAttack_baselines.ipynb)
- We provide our main experiment results under:  [`paper_experiments/modified_captions/`](modified_captions/)

### Evaluation
- We provide seperate evaluation scripts for MSCOCO and SNLI-VE datasets in [`paper_experiments/colab_notebooks/evaluation/`](paper_experiments/colab_notebooks/evaluation/)


# Citation
```bibtex
@inproceedings{ramshetty2023xmai,
    title={Cross-Modal Attribute Insertions for Assessing the Robustness of Vision-and-Language Learning},
    author={Ramshetty, Shivaen and Verma, Gaurav and Kumar, Srijan},
    booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2023}
}
```

# Acknowledgements
We also thank the authors and contributors of the following repositories:

```bibtex
@misc{yu2020buapt,
  author = {Yu, Zhou and Li, Jing and Luo, Tongan and Yu, Jun},
  title = {A PyTorch Implementation of Bottom-Up-Attention},
  howpublished = {\url{https://github.com/MILVLG/bottom-up-attention.pytorch}},
  year = {2020}
}
```

```bibtex
@inproceedings{dou2022meter,
  title={An Empirical Study of Training End-to-End Vision-and-Language Transformers},
  author={Dou, Zi-Yi and Xu, Yichong and Gan, Zhe and Wang, Jianfeng and Wang, Shuohang and Wang, Lijuan and Zhu, Chenguang and Zhang, Pengchuan and Yuan, Lu and Peng, Nanyun and Liu, Zicheng and Zeng, Michael},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022},
  url={https://arxiv.org/abs/2111.02387},
}
```

```bibtex
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

```bibtex
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
