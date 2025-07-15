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
    if domain == 'nuur':
        pt_dir = 'real_latent'
        preview_dir = 'static/previews'
    elif domain.startswith('skins_'):
        pt_dir = os.path.join('real_latent', domain.replace('skins_', '') + '_skin')
        preview_dir = os.path.join('static/previews', domain.replace('skins_', '') + '_skin')
    else:
        raise ValueError(f"Invalid domain: {domain}")

    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
    texture_items = [{"index": i, "img": f"{preview_dir}/{f}.png", "filename": f, "domain": domain} for i, f in enumerate(pt_files)]

    if len(texture_items) > 50:
        texture_items = random.sample(texture_items, 50)

    return render(request, 'home.html', {"texture_items": texture_items, "domain": domain})

def edit_texture(request, domain, index):
    pt_dir = 'real_latent' if domain == 'nuur' else os.path.join('real_latent', domain.replace('skins_', '') + '_skin')
    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
    filename = pt_files[int(index)]

    _, sim_glossy, sim_matte = run_inference(filename, method="none", strength=0, pt_dir=pt_dir)
    _, sim_rough, sim_smooth = run_inference_roughness(filename, method="none", strength=0, pt_dir=pt_dir)
    # In edit_texture() in views.py
    tone = domain.replace('skins_', '') if domain.startswith('skins_') else None
    if domain == 'nuur':
        preview_path = f"previews"
    else:
        preview_path = f"previews/{tone}_skin"
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

    pt_dir = 'real_latent' if domain == 'nuur' else os.path.join('real_latent', domain.replace('skins_', '') + '_skin')
    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
    filename = pt_files[index]

    img_url, sim_glossy, sim_matte = run_inference(filename, method, strength, pt_dir)
    return JsonResponse({
        "img_url": img_url,
        "sim_glossy": round(sim_glossy, 3),
        "sim_matte": round(sim_matte, 3)
    })

def update_image_rough(request):
    index = int(request.GET.get("index"))
    method = request.GET.get("method")
    strength = float(request.GET.get("strength"))
    domain = request.GET.get("domain")

    pt_dir = 'real_latent' if domain == 'nuur' else os.path.join('real_latent', domain.replace('skins_', '') + '_skin')
    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith('.pt')])
    filename = pt_files[index]

    img_url, sim_rough, sim_smooth = run_inference_roughness(filename, method, strength, pt_dir)
    return JsonResponse({
        "img_url": img_url,
        "sim_rough": round(sim_rough, 3),
        "sim_smooth": round(sim_smooth, 3)
    })

import csv

ANSWER_CSV = "static/answers.csv"

def submit_answer(request):
    index = request.GET.get("index")
    domain = request.GET.get("domain")
    attr = request.GET.get("attribute")
    value = request.GET.get("value")

    key = f"{domain}_{index}"
    updated = False
    rows = []

    # Load old answers
    if os.path.exists(ANSWER_CSV):
        with open(ANSWER_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Check for existing entry
    for row in rows:
        if row["key"] == key:
            row[attr] = value
            updated = True
            break

    # Add new entry if not found
    if not updated:
        new_row = {"key": key, "glossier": "", "matte": "", "rough": "", "smooth": ""}
        new_row[attr] = value
        rows.append(new_row)

    # Write back
    with open(ANSWER_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["key", "glossier", "matte", "rough", "smooth"])
        writer.writeheader()
        writer.writerows(rows)

    return JsonResponse({"message": f"Saved answer: {attr} → {value}"})
