# views.py
import os
import random
from django.shortcuts import render
from django.http import JsonResponse
from .inference import run_inference

# -------------------------
# Dataset selection
# -------------------------
def select_dataset(request):
    return render(request, "select_dataset.html")

def select_skin_type(request):
    return render(request, "select_skin_type.html")

# -------------------------
# Helpers
# -------------------------
def _resolve_dirs(domain: str):
    if domain == 'nuur':
        pt_dir = 'real_latent_9475'
        preview_dir = 'static/previews/nuur_9475'
    elif domain.startswith('skins_'):
        tone = domain.replace('skins_', '')
        pt_dir = os.path.join('real_latent_9475', f"{tone}_skin")
        preview_dir = os.path.join('static/previews/nuur_9475', f"{tone}_skin")
    elif domain == 'generated':
        pt_dir = 'real_latent_9475/generated'
        preview_dir = 'static/previews/generated_9475'
    else:
        raise ValueError(f"Invalid domain: {domain}")
    return pt_dir, preview_dir

# -------------------------
# Dataset browser
# -------------------------
def home(request, domain):
    # cleanup tmp
    tmp_dir = 'static/tmp'
    if os.path.exists(tmp_dir):
        for f in os.listdir(tmp_dir):
            f_path = os.path.join(tmp_dir, f)
            if os.path.isfile(f_path):
                os.remove(f_path)

    pt_dir, preview_dir = _resolve_dirs(domain)

    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
    texture_items = [
        {"index": i, "img": f"{preview_dir}/{f}.png", "filename": f, "domain": domain}
        for i, f in enumerate(pt_files)
    ]
    if len(texture_items) > 50:
        texture_items = random.sample(texture_items, 50)

    return render(request, 'home.html', {"texture_items": texture_items, "domain": domain})

# -------------------------
# Texture editor
# -------------------------
def edit_texture(request, domain, index):
    pt_dir, preview_dir = _resolve_dirs(domain)
    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
    filename = pt_files[int(index)]

    # --- Baseline from original (method="none", strength=0) ---
    # We want similarities for ALL attributes + a single baseline histogram.
    attr_list = ["glossy", "matte", "rough", "smooth", "regular", "random", "coarse", "fine"]
    baseline = {}
    baseline_hist_url = None
    baseline_skew = None

    # Use glossy call to produce the baseline histogram (any attr works; image is unchanged)
    (
        _, sim_attr1, sim_attr2, _, _, _, _, hist_url, skew_val
    ) = run_inference(filename, method="none", strength=0, pt_dir=pt_dir, attr="glossy")
    baseline_hist_url = hist_url
    baseline_skew = skew_val
    baseline["glossy"] = float(sim_attr1)
    baseline["matte"]  = float(sim_attr2)

    # Gather remaining attributes
    for a in ["rough", "regular", "coarse"]:
        _, s1, s2, _, _, _, _, _, _ = run_inference(filename, method="none", strength=0, pt_dir=pt_dir, attr=a)
        # map each pair back
        if a == "rough":
            baseline["rough"]  = float(s1)
            baseline["smooth"] = float(s2)
        elif a == "regular":
            baseline["regular"] = float(s1)
            baseline["random"]  = float(s2)
        elif a == "coarse":
            baseline["coarse"] = float(s1)
            baseline["fine"]   = float(s2)

    # preview_path for template usage
    if domain == 'nuur':
        preview_path = f"previews/nuur_9475"
    elif domain.startswith('skins_'):
        tone = domain.replace('skins_', '')
        preview_path = f"previews/nuur_9475/{tone}_skin"
    else:
        preview_path = 'previews/generated_9475'

    return render(request, "edit.html", {
        "index": index,
        "filename": filename,
        "domain": domain,
        "preview_path": preview_path,
        # Rows:
        "gloss_methods": ["bs", "scurve", "clip"],  # first row
        "rough_methods": ["bs", "clip"],            # second row
        "pattern_attrs": ["regular", "random", "coarse"],  # third row (clip)
        # Baseline (original) info:
        "baseline_scores": {k: round(v, 3) for k, v in baseline.items()},
        "baseline_hist_url": baseline_hist_url,
        "baseline_skew": baseline_skew,
    })

# -------------------------
# Update image (unified)
# -------------------------
def update_image(request):
    index = int(request.GET.get("index"))
    method = request.GET.get("method")       # "bs" | "scurve" | "clip" | "none"
    strength = float(request.GET.get("strength"))
    domain = request.GET.get("domain")
    attr = request.GET.get("attr", "glossy") # which attribute pair to score against

    pt_dir, _ = _resolve_dirs(domain)
    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
    filename = pt_files[index]

    (
        img_url, sim_attr1, sim_attr2, sim_img, stsim, sw, nat,
        hist_url, skew_val
    ) = run_inference(filename, method, strength, pt_dir, attr)

    return JsonResponse({
        "img_url": img_url,
        "sim_attr1": round(float(sim_attr1), 3),
        "sim_attr2": round(float(sim_attr2), 3),
        "sim_img": round(float(sim_img), 3),
        "stsim": round(float(stsim), 3),
        "sw": round(float(sw), 3),
        "nat": bool(nat),
        "hist_url": hist_url,
        "skew": float(skew_val),
    })

# -------------------------
# CSV answers API (unchanged)
# -------------------------
import csv

ANSWER_CSV = "static/answers.csv"
CSV_FIELDS = [
    "key",
    "glossier_possible", "matte_possible", "rough_possible", "smooth_possible",
    "best_glossiness_method", "best_roughness_method"
]
ALIAS_MAP = {
    "glossier": "glossier_possible",
    "matte": "matte_possible",
    "rough": "rough_possible",
    "smooth": "smooth_possible",
    "glossiness": "best_glossiness_method",
    "roughness": "best_roughness_method",
}

def submit_answer(request):
    index = request.GET.get("index")
    domain = request.GET.get("domain")
    attr = request.GET.get("attribute")
    value = request.GET.get("value")

    key = f"{domain}_{index}"
    target_field = ALIAS_MAP.get(attr)
    if not target_field:
        return JsonResponse({"error": f"Invalid attribute: {attr}"}, status=400)

    rows = []
    if os.path.exists(ANSWER_CSV):
        with open(ANSWER_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    updated = False
    for row in rows:
        if row["key"] == key:
            row[target_field] = value
            updated = True
            break

    if not updated:
        new_row = {field: "" for field in CSV_FIELDS}
        new_row["key"] = key
        new_row[target_field] = value
        rows.append(new_row)

    with open(ANSWER_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return JsonResponse({"message": f"Saved {target_field} = {value}"})
