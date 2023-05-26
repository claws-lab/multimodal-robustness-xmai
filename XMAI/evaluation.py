from __future__ import division
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score

import os
import argparse
from io import BytesIO
import base64
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import clip
from tqdm import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input_csv',
        type=str,
        required=True,
        help="Path to the input csv file"
    )
    parser.add_argument(
        '-t', '--text_columns',
        type=str,
        default=["text", "augmented"],
        nargs=2,
        help="Name of CSV columns containing text inputs"
    )
    parser.add_argument(
        '-f', '--file_column',
        type=str,
        default="file",
        help="Name of CSV column containing paths or names of image files"
    )
    parser.add_argument(
        '-d', '--image_dir',
        type=str,
        default=None,
        help="Path to image dir, not necessary if csv contains full paths"
    )

    args = parser.parse_args()

    assert os.path.isfile(args.input_csv)
    assert args.image_dir is None or os.path.isdir(args.image_dir)

    return args


def MRR(similarities):
    """
    Compute MRR for a similarity matrix.
    """
    mrr = 0
    for cap in range(len(similarities)):
        # Since 1-1 image-text pair, retrieve similarity value of the actual pair
        correct_sim = similarities[cap][cap]
        # Get indices in descending order     
        sort_image_sim = np.sort(similarities[cap])[::-1]
        # Use highest position of actual pair's similarity as its rank
        mrr += 1/(np.where(sort_image_sim == correct_sim)[0][0] + 1)

    return mrr/len(similarities)


def avg_bleu(og_caps, mod_caps):
    """
    Compute mean BLEU score between original and augmented texts.
    """
    smoother = SmoothingFunction()

    scores = []
    for og, mod in zip(og_caps, mod_caps):
        og = word_tokenize(og)
        mod = word_tokenize(mod)
        scores.append(bleu([og], mod, smoothing_function=smoother.method4, auto_reweigh=True))

    return np.mean(scores)


def avg_meteor(og_hyps, mod_hyps):
    """
    Compute mean METEOR score between original and augmented texts.
    """
    scores = []
    for og, mod in zip(og_hyps, mod_hyps):
        og = word_tokenize(og)
        mod = word_tokenize(mod)
        scores.append(meteor_score([og], mod, gamma=0.5))

    return np.mean(scores)


# Compute mean sentence similarity over all caption pairs
def mean_relevancy(sent_model, original, augmented, batch_size=256):
    """
    Compute mean text similarities between original and augmented texts.
    """
    rels = []
    for i in range(0, len(original), batch_size):
        with torch.no_grad():
            og_embs = sent_model.encode(original[i:i+batch_size])
            aug_embs = sent_model.encode(augmented[i:i+batch_size])

        rels += torch.diag(util.cos_sim(og_embs, aug_embs)).tolist()
    
    # return mean and std
    return np.mean(rels), np.std(rels)


def build_features(clip_model, imgs, og_caps, mod_caps):
    """
    Compute image, original text, and augmented text embeddings with CLIP model.
    """
    image_input = torch.tensor(np.stack(imgs)).cuda()
    orig_text_tokens = clip.tokenize(["This is " + desc for desc in og_caps], truncate=True).cuda()
    mod_text_tokens = clip.tokenize(["This is " + desc for desc in mod_caps], truncate=True).cuda()

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()
        orig_text_features = clip_model.encode_text(orig_text_tokens).float()
        mod_text_features = clip_model.encode_text(mod_text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    orig_text_features /= orig_text_features.norm(dim=-1, keepdim=True)
    mod_text_features /= mod_text_features.norm(dim=-1, keepdim=True)

    return image_features, orig_text_features, mod_text_features


def batch_experiment(clip_model, preprocess, og_texts, aug_texts, files, im_dir, batch_size=1000):
    """
    Compute similarity between image-text pairs for both original and augmented texts. This function
    can be used to produce MRRs for batches as well.
    """
    img_og_rel = []
    img_mod_rel = []
    for i in tqdm(range(0, len(og_texts), batch_size)):
        images = []
        # Ignore batches that are not "full"
        if len(og_texts) - i < batch_size:
            continue
        for filename in files[i:i+batch_size]:
            try:
                img = Image.open(os.path.join(im_dir, filename)).convert("RGB")
            except :
                try:
                    img = Image.open(BytesIO(base64.urlsafe_b64decode(filename)))
                except:
                    print("ERROR: Can't find/read image when computing CLIP dissimilarity.")
                    raise RuntimeError
            images.append(preprocess(img))

        original = og_texts[i:i+batch_size]
        augmented = aug_texts[i:i+batch_size]

        image_features, orig_text_features, mod_text_features = build_features(clip_model, images, original, augmented)
        image_features = image_features
        orig_text_features = orig_text_features
        mod_text_features = mod_text_features

        orig_image_sim = orig_text_features @ image_features.T
        mod_image_sim = mod_text_features @ image_features.T
        orig_image_sim = orig_image_sim.cpu().numpy()
        mod_image_sim = mod_image_sim.cpu().numpy()

        img_og_rel += list(np.diag(orig_image_sim))
        img_mod_rel += list(np.diag(mod_image_sim))
    
    print(f"Original Caption Image Relevance - Mean: {np.mean(img_og_rel)}, Std: {np.std(img_og_rel)}")
    print(f"Modified Caption Image Relevance - Mean: {np.mean(img_mod_rel)}, Std: {np.std(img_mod_rel)}")


if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('wordnet')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    sent_model = SentenceTransformer('all-mpnet-base-v2').to(device)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    data_df = pd.read_csv(args.input_csv)

    og_texts = data_df[args.text_columns[0]].to_list()
    aug_texts = data_df[args.text_columns[1]].to_list()

    print("Comparing original and augmented texts ...")
    print("Mean BLEU scores:", avg_bleu(og_texts, aug_texts))
    print("Mean METEOR scores:", avg_meteor(og_texts, aug_texts))
    print("Mean sentence similarity:", mean_relevancy(sent_model, og_texts, aug_texts))

    print("Comparing texts to images...")
    files = data_df[args.file_column]
    batch_experiment(clip_model, preprocess, og_texts, aug_texts, files, args.image_dir)
