import argparse
from io import BytesIO
import base64
from PIL import Image
import os
from ast import literal_eval

import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import clip
import nltk
import spacy
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util
from flair.models import SequenceTagger

from utils import pos_objects, tokens_to_text, combine_token_pieces, get_sw_tokens


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input_csv',
        type=str,
        required=True,
        help="Path to the input csv file"
    )
    parser.add_argument(
        '-t', '--text_column',
        type=str,
        default="text",
        help="Name of CSV column containing text input"
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
    parser.add_argument(
        '-o', '--output_csv',
        type=str,
        default="./modified_text/augmented.csv",
        help="Path to output csv file"
    )
    parser.add_argument(
        "--mask_model",
        type=str,
        default="bert-base-cased",
        help="Which HuggingFace model to use for [MASK] prediction"
    )
    parser.add_argument(
        "--mask_method",
        type=str,
        default="mix",
        choices=['direct', 'sim', 'mix'],
        help="Which masking method to use of: ['direct', 'sim', 'mix']"
    )
    parser.add_argument(
        "--lambda_a",
        type=int,
        default=1,
        help="Parameter controlling influence of LM probability on chosen token, higher means more influence"
    )
    parser.add_argument(
        "--lambda_b",
        type=int,
        default=1,
        help="Parameter controlling influence of attribute similarity on chosen token, higher means more influence"
    )
    parser.add_argument(
        "--lambda_c",
        type=int,
        default=1,
        help="Parameter controlling influence of clip dissimilarity on chosen token, higher means more influence"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of tokens to consider for each [MASK]"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for token-attribute comparison"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print outputs of each step"
    )

    args = parser.parse_args()

    assert os.path.isfile(args.input_csv)
    assert args.image_dir is None or os.path.isdir(args.image_dir)

    return args


def transform_similar_objects(sen_model, tagger, orig_text, img_objects, threshold=0.7):
    """
    Add [MASK] tokens in front of objects, where objects in text are above some
    similarity threshold to those detected.

    Args:
        sen_model: Sentence transformer model
        tagger: POS tagging model
        orig_text: Text to augment
        img_objects: List of objects detected in the corresponding image
        threshold: Similarity threshold determining whether object in text is semantic
            close enough to an object in 'img_objects'
    """
    objs, obj_positions = pos_objects(orig_text, tagger) # Find all noun-type objects
    cap_objs = sen_model.encode(objs)
    img_objs = sen_model.encode(list(img_objects))
    
    if cap_objs.shape[-1] != img_objs.shape[-1]:
        return orig_text, []
    sims = util.cos_sim(cap_objs, img_objs) # sims shape: N_cap_objs, N_img_objs
    thresh_pos_r, thresh_pos_c = torch.where(sims >= threshold)
    
    obj_idx = {}
    text = orig_text
    shift = 0 # Number of masks already introduced
    for r, c in zip(thresh_pos_r, thresh_pos_c):
        if objs[r] in obj_idx:
            obj_idx[objs[r]] += [c.item()]
            continue
        # 7*shift since "[MASK] " is 7 characters, note the trailing space
        text = "".join([text[:obj_positions[r] + 7*shift], "[MASK] ", text[obj_positions[r] + 7*shift:]])
        shift += 1
        obj_idx[objs[r]] = [c.item()]
    return text, obj_idx


def mask_text(text, objects):
    """
    Insert [MASK] tokens in front of exactly matched objects in the given text.
    If no exact match exists, text will not be modified.

    Args:
        text: Text to augment
        objects: List of objects detected in the corresponding image 
    """
    # Store every matched object and the indices the matches occur in the original text
    unique = {}
    for i in range(len(objects)):
        # If object already matched, then don't repeat the check
        if objects[i] in unique:
            unique[objects[i].split()[0]].append(i)
            continue
        
        idx = text.find(objects[i])
        if idx > 0 and text[idx-1] == " ":
            # Check if match is exactly the object => "goal" shouldn't match with "goalie"
            if (idx+len(objects[i]) < len(text) and text[idx+len(objects[i])] == " ") or idx+len(objects[i]) == len(text):
                # Make sure we don't repeat multiple masks if one already exists in this position for some reason
                if idx >= 7 and text[idx-7:idx-1] == "[MASK]":
                    continue
                unique[objects[i].split()[0]] = [i]
                text = "".join([text[:idx], "[MASK] ", text[idx:]])
            
    return text, unique


# referenced https://gist.github.com/yuchenlin/a2f42d3c4378ed7b83de65c7a2222eb2
def mask_predictions(model, tokenizer, orig_tokens, m_idx, k):
    """
    Use LM to predict the top-k words for a given [MASK]. Return the k tokens
    and their probabilities.
    
    Args:
        model: LM for mask prediction.
        tokenizer: Tokenizer for given model.
        orig_tokens: List of tokens from text to augment
        m_idx: Mask index within 'orig_tokens'
        k: Number of tokens to return; top-k predicted tokens that are not
            stopwords, the same as the object, the same as preceeding 3 tokens,
            or [UNK] token.
    """
    pred_tokens = []
    token_probs = np.zeros((k, 1))
    
    orig_tokens.insert(m_idx, "[MASK]")
    obj = orig_tokens[m_idx + 1]
    adj_tokens = orig_tokens[m_idx-3:m_idx]

    indexed_tokens = tokenizer.convert_tokens_to_ids(orig_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to(device)

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, m_idx], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, 20, sorted=True) # Some of the matches may not be appropriate candidates, so gather more than k
    
    num_preds = 0
    for i, pred_idx in enumerate(top_k_indices):
        if pred_idx.item() not in set(sw_ids):
            token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
            if token != obj and token not in adj_tokens and token != "[UNK]":
                pred_tokens.append(token)
                token_probs[num_preds] = top_k_weights[i].item()
              
                num_preds += 1
                if num_preds == k:
                    break
    
    return pred_tokens, token_probs


def token_similarity(preds, attrs):
    """
    Compute similarity matrix for predicted tokens and image attributes.

    Args:
        preds: List of predicted tokens
        attrs: List of attributes from image
    """
    if len(attrs) == 0:
        return np.zeros((len(preds), 1))

    sim_mat = np.zeros((len(preds), len(attrs)))
    for i, pred in enumerate(preds):
        pred_token = en_embeds(pred)
        for j, attr in enumerate(attrs):
            attr_token = en_embeds(attr)
            sim_mat[i, j] = pred_token.similarity(attr_token)

    return sim_mat


def clip_dissimilarity(clip_model, preprocess, texts, image_path):
    """
    Compute dissimilarity between cadidate texts and a given image.

    Args:
        clip_model: CLIP model
        preprocess: Preprocessing transforms
        texts: Candidate texts
        image_path: Path to or bytes of image
    """
    sims = np.array((len(texts), 1))

    try:
        image = preprocess(Image.open(image_path).convert("RGB"))
    except FileNotFoundError:
        try:
            image = preprocess(Image.open(BytesIO(base64.urlsafe_b64decode(image_path))).convert("RGB"))
        except:
            print("ERROR: Can't find/read image when computing CLIP dissimilarity.")
            raise RuntimeError


    image_input = torch.tensor(np.expand_dims(np.array(image), 0)).to(device)
    text_tokens = clip.tokenize(["This is " + desc for desc in texts], truncate=True).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()
        text_features = clip_model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    sims = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    
    return 1 - sims


def XMAI(
    args, 
    model,
    tokenizer,
    clip_model,
    preprocess,
    sent_model,
    tagger,
    orig_text,
    objects,
    attributes,
    image_file
):
    """
    Perform XMAI augmentation over given text with given objects and attributes. For each mask, begin by 
    performing masking of the n  text based on a selected object masking method. Then predicts most likely
    tokens for that mask using provided LM. Then, compares predicted tokens to 'attributes' and creates candidate
    texts for each token. These candidates are used for CLIP dissimilarity computation. Finally, scores for a given
    mask are aggregated and the max scoring token is chosen for the mask. This is repeated for all existing masks,
    returning an augmented text.

    Args:
        args: Arguments for XMAI configuration. Take a look at parse_args() for more details.
        model: LM for [MASK] token prediction.
        tokenizer: Tokenizer for LM.
        clip_model: CLIP model
        preprocess: Preprocessing transforms
        sent_model: Sentence transformer
        tagger: POS tagging model
        orig_text: Text to augment
        objects: List of objects detected in image
        attributes: List of attributes detected in image
        image_file: Path to or bytes if image
    """

    # Add buffer
    orig_tokens = ['']  + tokenizer.tokenize(orig_text) + ['']
    
    # Create text with masks in front of relevant objects
    if args.verbose: print("Original Text:\n", orig_text, end="\n\n")

    if args.mask_method == "direct":
        masked_text, object_indices = mask_text(orig_text, objects) # Mask direct matches
    elif args.mask_method == "sim":
        if args.verbose: print("Finding similar objects with threshold = %f"%args.threshold)
        masked_text, object_indices = transform_similar_objects(sent_model, tagger, orig_text, objects, args.threshold) # Mask words similar to detected objects within threshold
    else:
        masked_text, object_indices = mask_text(orig_text, objects) # Mask direct matches
        if len(object_indices) == 0:
            if args.verbose: print("Finding similar objects with threshold = %f"%args.threshold)
            masked_text, object_indices = transform_similar_objects(sent_model, tagger, orig_text, objects, args.threshold) # Mask words similar to detected objects within threshold
        
    if args.verbose: print(object_indices)

    tokens = [''] + tokenizer.tokenize(masked_text) + ['']
    if args.verbose: print(tokens)

    # Combine pieced tokens so we can find objects within token list
    object_tokens = combine_token_pieces(tokens)
    if args.verbose: print(object_tokens)

    masked_indices = [i for i in range(len(tokens)) if tokens[i] == '[MASK]']

    for m_idx in masked_indices:
        
        score = np.zeros((args.k, 1))
        
        # Part A: Get mask token predictions and associated probabilities -> a = probs
        pred_tokens, probs = mask_predictions(model, tokenizer, orig_tokens, m_idx, args.k)
        if args.verbose:
            print("Part A: ")
            print("predicted tokens for object %s: %s"%(object_tokens[m_idx+1], str(pred_tokens)))
            print("probabilites:\n", probs)
            print("-----------------------------------------------------------")
        p_max = np.max(probs)
        p_min = np.min(probs)
        if p_min == p_max:
              p_min = 0
        score += args.lambda_a * ((probs - p_min) / max(p_max - p_min, 1e-12))
        score = score[:len(pred_tokens)]

        # Part B: Get similarities between predicted words and attributes -> b = lambda_b * max(sim_matrix, axis=1)
        if object_tokens[m_idx+1] in object_indices:
            attrs = [attributes[a] for a in object_indices[object_tokens[m_idx+1]]]
        else:
            attrs = []
        sim_mat = token_similarity(pred_tokens, attrs)
        if args.verbose:
            print("Part B: ")
            print("relevant attributes for object %s:"%(object_tokens[m_idx+1]), attrs)
            print("similarity matrix:\n", sim_mat)
            print("-----------------------------------------------------------")
        token_sims = np.max(sim_mat, axis=1, keepdims=True)
        try:
            t_max = np.max(token_sims)
            t_min = np.min(token_sims)
            if t_min == t_max:
                t_min = 0
            score += args.lambda_b * ((token_sims - t_min) / max(t_max - t_min, 1e-12))
        except ValueError:
            score += 0
        
        # Part C: Create modified texts with each token and compute CLIP dissimilarity scores with corresponding image
        sample_texts = []
        for t, tok in enumerate(pred_tokens):
            tokens[m_idx] = tok
            sentence = tokens_to_text(tokens)
            sample_texts.append(sentence)
        
        image_path = os.path.join(args.image_dir, image_file)
        clip_sims = clip_dissimilarity(clip_model, preprocess, sample_texts, image_path)
        if args.verbose:
            print("Part C: ")
            for s in sample_texts:
                print(s)
            print("clip dissimilarities:\n", clip_sims)
            print("-----------------------------------------------------------")
        try:
            c_max = np.max(clip_sims)
            c_min = np.min(clip_sims)
            if c_min == c_max:
                c_min = 0
            score += args.lambda_c * ((clip_sims - c_min) / max(c_max - c_min, 1e-12))
        except ValueError:
            score += 0

        # Rescale scores
        score = score / max((abs(args.lambda_a) + abs(args.lambda_b) + abs(args.lambda_c)), 1)

        # Pick max scoring word
        if args.verbose:
            print("Final Results:")
            for p in range(len(pred_tokens)):
                print(pred_tokens[p], score[p])
            print("-----------------------------------------------------------", end="\n\n")

        if len(score) > 0:
            tokens[m_idx] = pred_tokens[np.argmax(score)]
        else:
            tokens[m_idx] = ""
    
    output_sentence = tokens_to_text(tokens)
    
    return output_sentence


def set_globals(tokenizer):
    """
    Initialize globals for processing functions and XMAI.
    """
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    global en_embeds
    en_embeds = spacy.load('en_core_web_md')

    global sw_ids
    _, sw_ids = get_sw_tokens(tokenizer)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.mask_model)
    set_globals(tokenizer)

    model = AutoModelForMaskedLM.from_pretrained(args.mask_model).to(device)
    model.eval()
    
    tagger = SequenceTagger.load("flair/pos-english-fast").to(device)

    sent_model = SentenceTransformer('bert-base-nli-mean-tokens').to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    augmented_text = []
    data_df = pd.read_csv(args.input_csv)

    for text, objects, attributes, image_file in \
        tqdm(
            zip(
                data_df[args.text_column],
                data_df['objects'],
                data_df['attributes'],
                data_df[args.file_column]
            ),
            position=0,
            total=len(data_df)
        ):

        augmented_text.append(
            XMAI(
                args,
                model,
                tokenizer,
                clip_model,
                preprocess,
                sent_model,
                tagger,
                text,
                literal_eval(objects),
                literal_eval(attributes),
                image_file
            )
        )

    # Save augmented text in new column.
    # Note this will replace 'augmented' column if it already exists
    data_df['augmented'] = augmented_text

    if not os.path.exists(args.output_csv):
        os.mkdir(os.path.dirname(args.output_csv))
    
    data_df.to_csv(args.output_csv, index=False)   

    return


if __name__ == "__main__":
    args = parse_args()

    nltk.download('stopwords')

    main(args)
