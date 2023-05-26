import string
from nltk.corpus import stopwords
from flair.data import Sentence


def pos_objects(sent, tagger):
    """
    Find objects in text by locating nouns or noun phrases.
    """
    sentence = Sentence(sent)
    tagger.predict(sentence)
    objects = []
    positions = []
    for entity in sentence:
        if entity.get_label().value.startswith("NN"):
            clean = ''.join([c for c in sent[entity.start_position:entity.end_position] if c.isalpha()])
            objects.append(clean)
            positions.append(entity.start_position)
    return objects, positions


def tokens_to_text(tokens, special=True):
    """
    Convert list of tokens to a single string, accounting for special tokens
    if 'special' is True.
    """
    sentence = ""
    start = 1
    end = -1
    if not special:
      start = 0
      end = len(tokens)
    apostrophe = False
    for tok in tokens[start:end]:
        if tok.startswith("##"):
            sentence += tok[2:]
        elif tok == "":
            continue
        elif tok in string.punctuation:
            if tok == "'":
                apostrophe = True
            sentence += tok
        elif apostrophe:
            sentence += tok
            apostrophe = False
        else:
            sentence += " " + tok
    return sentence[1:]


def combine_token_pieces(tokens):
    """
    Combine tokens that are broken by tokenization.
    Useful when fetching attributes for objects, since objects may be broken
    by tokenization in text but not in the provided inputs.
    """
    new_tokens = []
    i = 0
    parts = 0
    for tok in tokens:
        if type(tok) != int and tok.startswith("##"):
            new_tokens[i-1] += tok[2:]
            parts += 1
        else:
            if parts > 0:
                for _ in range(parts):
                    new_tokens.append(new_tokens[i-1])
                    i += 1
                parts = 0
            new_tokens.append(tok)
            i += 1
    return new_tokens


def get_sw_tokens(tokenizer):
    """
    Tokenize stopwords with same model as that performing the mask prediction.
    """
    sw_nltk = stopwords.words('english')
    sw_nltk = sw_nltk + [pun for pun in string.punctuation]
    sw_tokens = tokenizer.tokenize(" ".join(sw_nltk))
    sw_ids = tokenizer.convert_tokens_to_ids(sw_tokens)

    return sw_tokens, sw_ids