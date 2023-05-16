## Installation

```
conda create -n "xmai" python=3.8 -y
conda activate xmai
pip install -r requirements.txt

python -m spacy download en_core_web_md 
```
Will neeed to modify requirements depending on your CUDA version or if you plan on only using CPU.

## Augmentation

To augment your own data:
```
# Using default parameters and models
python xmai_generate.py \
    --input_csv "../paper_experiments/processed_data/coco_objs_attrs_captions.csv" \
    --output_csv "./modified_data/augmented.csv" \
    --text_column "original" \ 
    --file_column "file" \
    --image_dir d "../../data/coco_val17"
```

## Evaluation

We provide task-agnostic metrics is `evaluation.py`, however for task specific results we recommend taking a look at our notebooks: [retrieval](paper_experiments/colab_notebooks/evaluation/CLIP_MSCOCO.ipynb) or [classification](paper_experiments/colab_notebooks/evaluation/METER_SNLI_VE.ipynb)

```
python evaluation.py \
    --input_csv "../paper_experiments/modified_captions/MSCOCO/XMAI/mix_modified_thresh-07_a-1_b-5_c-5_k-3.csv" \
    --text_columns "original" "augmented" \
    --file_column "file" \
    --image_dir "../../data/coco_val17"
```