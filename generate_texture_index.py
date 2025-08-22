import os
import json

# Folder containing all .pt files and subfolders
latent_dir = "real_latent_9475"
output_json = "gloss_editor/texture_index.json"

# Ensure the folder exists
assert os.path.exists(latent_dir), f"Folder not found: {latent_dir}"

# Create index mapping
index_map = {}
index = 0

# First, process top-level .pt files in real_latent/
for fname in sorted(os.listdir(latent_dir)):
    fpath = os.path.join(latent_dir, fname)
    if os.path.isfile(fpath) and fname.endswith(".pt"):
        index_map[str(index)] = fname
        index += 1

# Then, process .pt files inside subfolders
for subdir in sorted(os.listdir(latent_dir)):
    subpath = os.path.join(latent_dir, subdir)
    if os.path.isdir(subpath):
        for fname in sorted(os.listdir(subpath)):
            if fname.endswith(".pt"):
                rel_path = os.path.join(subdir, fname).replace("\\", "/")  # normalize for web use
                index_map[str(index)] = rel_path
                index += 1

# Save JSON
with open(output_json, "w") as f:
    json.dump(index_map, f, indent=2)

print(f"Indexed {len(index_map)} .pt files → saved to {output_json}")
