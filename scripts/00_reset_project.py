"""
00_reset_project.py
Resets and initializes Hinglish-NLP project structure.
Creates required directories and baseline integrity manifest.
Author: Yuvraj Arora
"""

import os, json, hashlib, shutil

# Define folder structure
folders = [
    "data/human_annotated", "data/noisy_synthetic", "data/noisy_humanlike",
    "models/mbert", "models/canine", "models/byt5",
    "results/logs", "results/metrics", "results/charts", "configs", "scripts"
]

for f in folders:
    os.makedirs(f, exist_ok=True)

# Delete any leftover cache or temporary model data
for root, _, files in os.walk("models"):
    for file in files:
        if file.endswith((".pt", ".bin", ".safetensors")):
            os.remove(os.path.join(root, file))

# Generate integrity manifest
manifest = {}
for path, _, files in os.walk("."):
    for f in files:
        if f.endswith((".py", ".tsv", ".json")):
            p = os.path.join(path, f)
            with open(p, "rb") as fh:
                manifest[p] = hashlib.sha256(fh.read()).hexdigest()

json.dump(manifest, open("integrity_manifest.json", "w"), indent=2)
print("✅ Project reset complete and integrity manifest created.")
