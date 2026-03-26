"""
02_generate_noise.py (UPGRADED HUMANLIKE VERSION)
Realistic Hinglish noise generator with low/medium/high severity.

Author: Yuvraj Arora + ChatGPT Upgrade
"""

import pandas as pd
import random
import os
from pathlib import Path

# ------------------------------------------------------------
#  PHONETIC + SLANG + COMMON HINGLISH NOISE MAPS
# ------------------------------------------------------------
PHONETIC_MAP = {
    "kya": ["kia", "kyaaa", "kay", "ky"],
    "hai": ["he", "hy", "h"],
    "nahi": ["nhi", "nai", "nahe", "naahi"],
    "mera": ["mra", "meraa", "meyra"],
    "mat": ["matt", "mt"],
    "jaana": ["jana", "jaanaaa"],
    "haan": ["han", "haaan", "haannn"],
    "acha": ["accha", "achaa", "achaah"]
}

SLANG_MAP = {
    "please": ["plz", "pls", "plzzz"],
    "because": ["cuz", "bcuz"],
    "message": ["msg"],
    "tomorrow": ["tmrw"],
    "love": ["luv"],
    "you": ["u"],
    "are": ["r"]
}

# keyboard-neighbour mistakes (QWERTY)
KEYBOARD_NEIGHBORS = {
    "a": ["s", "q", "z"],
    "s": ["a", "d", "x"],
    "d": ["s", "f", "e"],
    "i": ["o", "u", "k"],
    "o": ["i", "p", "l"],
    "m": ["n", "j"],
    "n": ["m", "h"]
}

EMOJIS = ["😂", "😭", "🔥", "😅", "💀", "😎", "👍", "💯", "❤️", "🤦"]

PUNCT_NOISE = ["??", "!!!", "?!", "...", "!?!" ]


# ------------------------------------------------------------
#  BASIC RANDOM FUNCTIONS
# ------------------------------------------------------------

def random_keyboard_error(ch: str):
    """Replace character with a keyboard-neighbor error."""
    if ch.lower() in KEYBOARD_NEIGHBORS:
        return random.choice(KEYBOARD_NEIGHBORS[ch.lower()])
    return ch

def add_typo(text: str):
    """Randomly apply keyboard typo or random typo."""
    if len(text) < 2:
        return text
    
    pos = random.randint(0, len(text) - 1)
    
    if random.random() < 0.6:
        # keyboard-neighbour error
        return text[:pos] + random_keyboard_error(text[pos]) + text[pos+1:]
    else:
        # insertion/substitution/deletion
        char = random.choice("abcdefghijklmnopqrstuvwxyz")
        action = random.choice(["insert", "substitute", "delete"])
        
        if action == "insert":
            return text[:pos] + char + text[pos:]
        elif action == "substitute":
            return text[:pos] + char + text[pos+1:]
        else:
            return text[:pos] + text[pos+1:]

def repeat_chars(text: str):
    if not text:
        return text
    pos = random.randint(0, len(text)-1)
    return text[:pos] + text[pos]*random.randint(2,4) + text[pos+1:]

def random_case(text: str):
    return ''.join(
        c.upper() if random.random() < 0.3 else c.lower()
        for c in text
    )

def inject_emoji(text: str):
    pos = random.randint(0, len(text))
    return text[:pos] + " " + random.choice(EMOJIS) + " " + text[pos:]

def add_punctuation_noise(text: str):
    return text + random.choice(PUNCT_NOISE)

def add_whitespace_noise(text: str):
    if len(text) < 4:
        return text
    pos = random.randint(1, len(text)-2)
    return text[:pos] + " " + text[pos:]


# ------------------------------------------------------------
#  PHONETIC / SLANG / LEXICAL CORRUPTION
# ------------------------------------------------------------
def apply_phonetic_noise(text: str):
    words = text.split()
    new_words = []
    for w in words:
        wl = w.lower()
        if wl in PHONETIC_MAP and random.random() < 0.5:
            new_words.append(random.choice(PHONETIC_MAP[wl]))
        else:
            new_words.append(w)
    return " ".join(new_words)

def apply_slang(text: str):
    words = text.split()
    new_words = []
    for w in words:
        wl = w.lower()
        if wl in SLANG_MAP and random.random() < 0.5:
            new_words.append(random.choice(SLANG_MAP[wl]))
        else:
            new_words.append(w)
    return " ".join(new_words)


# ------------------------------------------------------------
#  FINAL NOISE COMBINER
# ------------------------------------------------------------
def add_noise(text: str, level="medium"):
    text = str(text)

    strength = {
        "low":      0.12,
        "medium":   0.28,
        "high":     0.50
    }[level]

    # PHONETIC + SLANG
    if random.random() < strength:
        text = apply_phonetic_noise(text)
    if random.random() < strength:
        text = apply_slang(text)

    # CHAR LEVEL
    if random.random() < strength:
        text = add_typo(text)
    if random.random() < strength:
        text = repeat_chars(text)
    if random.random() < strength:
        text = random_case(text)
    if random.random() < strength:
        text = add_whitespace_noise(text)

    # EMOJI + PUNCT
    if random.random() < strength:
        text = inject_emoji(text)
    if random.random() < strength:
        text = add_punctuation_noise(text)

    return text.strip()


# ------------------------------------------------------------
#  DATASET GENERATION
# ------------------------------------------------------------
def create_noisy_dataset(src_dir, dest_dir, level):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        src_path = f"{src_dir}/{split}.tsv"
        dest_path = f"{dest_dir}/{split}.tsv"

        df = pd.read_csv(src_path, sep="\t")
        df["cs_query"] = df["cs_query"].apply(lambda x: add_noise(x, level))

        df.to_csv(dest_path, sep="\t", index=False)
        print(f"✅ Generated {level} noise → {dest_path} ({len(df)} rows)")


# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    SRC = "data/human_annotated"

    for level in ["low", "medium", "high"]:
        DEST = f"data/noisy_humanlike_{level}"
        print(f"\n--- Generating '{level.upper()}' noise dataset ---")
        create_noisy_dataset(SRC, DEST, level)
