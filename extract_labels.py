import json
import spacy
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

# Define color and shape vocab
COLOR_WORDS = {
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "gold", "silver", "beige", "tan", "cyan"
}

SHAPE_WORDS = {
    "circle", "circular", "round", "square", "rectangular", "rectangle",
    "triangular", "triangle", "hexagonal", "hexagon", "diamond", "dot",
    "striped", "pattern", "grid", "line", "wavy", "curved", "straight","pile"
}

# Generic "view" words that should NOT be labels
GENERIC_VIEW_WORDS = {
    "view", "close", "picture", "image", "photo", "photograph", "scene",
    "shot", "perspective", "angle", "focus", "texture", "surface", "object", "amount", "lot"
    }

import re

import re

def extract_labels_from_caption(caption, fname):
    """Extract object, color, and shape labels from both caption and filename."""

    # --- Caption analysis ---
    doc = nlp(caption.lower())
    nouns = [t.lemma_ for t in doc if t.pos_ == "NOUN" and len(t.text) > 2]
    adjectives = [t.lemma_ for t in doc if t.pos_ == "ADJ"]

    colors = [w for w in adjectives if w in COLOR_WORDS]
    shapes = [w for w in nouns + adjectives if w in SHAPE_WORDS]

    objects = [
        w for w in nouns + adjectives
        if w not in COLOR_WORDS
        and w not in SHAPE_WORDS
        and w not in GENERIC_VIEW_WORDS
    ]

    caption_labels = list(dict.fromkeys(colors + shapes + objects))

    # --- Filename analysis (using spaCy) ---
    base = fname.lower()
    base = re.sub(r'\.(png|jpg|jpeg|pt|json|txt)$', '', base)
    base = re.sub(r'[^a-z]+', ' ', base)

    fname_doc = nlp(base)
    fname_words = [
        t.lemma_ for t in fname_doc
        if t.pos_ in {"NOUN", "ADJ"}
        and len(t.text) > 2
        and t.lemma_ not in {"photo", "photos", "image", "images", "texture", "textures",
                             "picture", "img", "pix", "pixel", "pixels", "dataset", "sample", "file","fcbd","adf","oxrg", "color", "jpeg","text", "wildtexture"}
    ]

    # Combine caption + filename labels
    labels = list(dict.fromkeys(caption_labels + fname_words))
    labels = [w for w in labels if w not in GENERIC_VIEW_WORDS]
    return labels


def main(input_path="captions_G.json", output_path="captions_labels_G.json"):
    with open(input_path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    result = {}
    for fname, caption in captions.items():
        result[fname] = extract_labels_from_caption(caption, fname)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved extracted labels to {output_path}")

if __name__ == "__main__":
    main()
