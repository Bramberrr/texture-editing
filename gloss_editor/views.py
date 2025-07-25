# views.py
import os
import json
import random
from django.shortcuts import render
from django.http import JsonResponse
from .inference import run_inference, run_inference_roughness

# Dataset selection view
def select_dataset(request):
    return render(request, "select_dataset.html")

def select_skin_type(request):
    return render(request, "select_skin_type.html")

# Dataset browser view
def home(request, domain):
    tmp_dir = 'static/tmp'
    if os.path.exists(tmp_dir):
        for f in os.listdir(tmp_dir):
            f_path = os.path.join(tmp_dir, f)
            if os.path.isfile(f_path):
                os.remove(f_path)
    if domain == 'nuur':
        pt_dir = 'real_latent'
        preview_dir = 'static/previews'
    elif domain.startswith('skins_'):
        pt_dir = os.path.join('real_latent', domain.replace('skins_', '') + '_skin')
        preview_dir = os.path.join('static/previews', domain.replace('skins_', '') + '_skin')
    elif domain == 'generated':
        pt_dir = 'real_latent/generated'
        preview_dir = 'static/previews/generated'
    else:
        raise ValueError(f"Invalid domain: {domain}")

    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
    texture_items = [{"index": i, "img": f"{preview_dir}/{f}.png", "filename": f, "domain": domain} for i, f in enumerate(pt_files)]

    if len(texture_items) > 50:
        texture_items = random.sample(texture_items, 50)

    return render(request, 'home.html', {"texture_items": texture_items, "domain": domain})

def edit_texture(request, domain, index):
    if domain == 'nuur':
        pt_dir = 'real_latent'
    elif domain.startswith('skins_'):
        pt_dir = os.path.join('real_latent', domain.replace('skins_', '') + '_skin')
    elif domain == 'generated':
        pt_dir = 'real_latent/generated'
    else:
        raise ValueError(f"Invalid domain: {domain}")
    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
    filename = pt_files[int(index)]

    _, sim_glossy, sim_matte,_,_,_ = run_inference(filename, method="none", strength=0, pt_dir=pt_dir)
    _, sim_rough, sim_smooth,_,_,_ = run_inference_roughness(filename, method="none", strength=0, pt_dir=pt_dir)
    # In edit_texture() in views.py
    tone = domain.replace('skins_', '') if domain.startswith('skins_') else None
    if domain == 'nuur':
        preview_path = f"previews"
    elif domain.startswith('skins_'):
        preview_path = f"previews/{tone}_skin"
    elif domain == 'generated':
        preview_path = 'previews/generated'
    return render(request, "edit.html", {
        "index": index,
        "filename": filename,
        "domain": domain,
        "preview_path": preview_path,
        "methods": ["bs", "scurve", "clip"],
        "rough_methods": ["bs", "clip"],
        "original_scores": {
            "glossy": round(sim_glossy, 3),
            "matte": round(sim_matte, 3),
            "rough": round(sim_rough, 3),
            "smooth": round(sim_smooth, 3),
        }
    })

def update_image(request):
    index = int(request.GET.get("index"))
    method = request.GET.get("method")
    strength = float(request.GET.get("strength"))
    domain = request.GET.get("domain")

    if domain == 'nuur':
        pt_dir = 'real_latent'
    elif domain.startswith('skins_'):
        pt_dir = os.path.join('real_latent', domain.replace('skins_', '') + '_skin')
    elif domain == 'generated':
        pt_dir = 'real_latent/generated'
    else:
        raise ValueError(f"Invalid domain: {domain}")
    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
    filename = pt_files[index]

    img_url, sim_glossy, sim_matte, sim_img, stsim, sw = run_inference(filename, method, strength, pt_dir)
    return JsonResponse({
        "img_url": img_url,
        "sim_glossy": round(sim_glossy, 3),
        "sim_matte": round(sim_matte, 3),
        "sim_img": round(sim_img, 3),
        "stsim": round(stsim, 3),
        "sw": round(sw, 3)
    })

def update_image_rough(request):
    index = int(request.GET.get("index"))
    method = request.GET.get("method")
    strength = float(request.GET.get("strength"))
    domain = request.GET.get("domain")

    if domain == 'nuur':
        pt_dir = 'real_latent'
    elif domain.startswith('skins_'):
        pt_dir = os.path.join('real_latent', domain.replace('skins_', '') + '_skin')
    elif domain == 'generated':
        pt_dir = 'real_latent/generated'
    else:
        raise ValueError(f"Invalid domain: {domain}")
    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
    filename = pt_files[index]

    img_url, sim_rough, sim_smooth, sim_img, stsim, sw = run_inference_roughness(filename, method, strength, pt_dir)
    return JsonResponse({
        "img_url": img_url,
        "sim_rough": round(sim_rough, 3),
        "sim_smooth": round(sim_smooth, 3),
        "sim_img": round(sim_img, 3),
        "stsim": round(stsim, 3),
        "sw": round(sw, 3)
    })

import os
import csv
from django.http import JsonResponse

ANSWER_CSV = "static/answers.csv"
CSV_FIELDS = [
    "key",
    "glossier_possible", "matte_possible", "rough_possible", "smooth_possible",
    "best_glossiness_method", "best_roughness_method"
]

# Map from front-end field to CSV field
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
    attr = request.GET.get("attribute")     # e.g., 'glossiness' or 'glossier'
    value = request.GET.get("value")        # e.g., 'clip' or 'true'

    key = f"{domain}_{index}"
    target_field = ALIAS_MAP.get(attr)

    if not target_field:
        return JsonResponse({"error": f"Invalid attribute: {attr}"}, status=400)

    # Load existing rows
    rows = []
    if os.path.exists(ANSWER_CSV):
        with open(ANSWER_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Search for existing row
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

    # Save updated CSV
    with open(ANSWER_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return JsonResponse({"message": f"Saved {target_field} = {value}"})
