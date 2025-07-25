import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import os
from model.trainer import get_generator_kwargs
import dnnlib
import numpy as np
import torch
import torch.nn as nn
from model.s_transformation import S_Transformation
from model.band_sifting import band_sifting_editing
import open_clip
import json
import torch.nn.functional as F
from torchvision import models
OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
GENERATOR_CKPT = 'stylegan3-generator.pt'
SCALING_FACTOR = 1  # or 4, depending on how dense you want generated activations
CFG = 'stylegan3-r'
RES = 256
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Generator ---
G_kwargs = get_generator_kwargs(device=device)
generator = dnnlib.util.construct_class_by_name(**G_kwargs)
generator.load_state_dict(torch.load(GENERATOR_CKPT, map_location='cpu'), strict=False)
generator.eval().to(device)

clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
clip_model.eval().to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()
class STSIM_VGG(torch.nn.Module):
    def __init__(self, dim, grayscale=True):
        super(STSIM_VGG, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))
        self.C = 1e-10

        for param in self.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(dim[0], dim[1])
        if grayscale:
            self.chns = [1,64,128,256,512,512]
        else:
            self.chns = [3,64,128,256,512,512]
        # self.load_state_dict(torch.load('./pretrained_models/repeat4.pt'), strict=False)
        self.load_state_dict(torch.load('./feature_5900_noabsb.pt'), strict=False)
        # self.load_state_dict(torch.load('/data/zxq0322/Shift-Net/pretrained_models/feature_5900_noabsb.pt'), strict=False)
        # self.load_state_dict(torch.load('/data/zxq0322/PerceptualSimilarity/checkpoints/5900/1_net_.pth'), strict=False)

    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h

        coeffs = [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        f = []

        for c in coeffs:
            mean = torch.mean(c, dim=[2, 3])
            
            f.append(mean)
            
            var = torch.var(c, dim=[2, 3])
            f.append(var)
            c = c - mean.unsqueeze(-1).unsqueeze(-1)
            f.append(torch.mean(c[:, :, :-1, :] * c[:, :, 1:, :], dim=[2, 3]) / (var + self.C))
            f.append(torch.mean(c[:, :, :, :-1] * c[:, :, :, 1:], dim=[2, 3]) / (var + self.C))
        # import pdb;pdb.set_trace()
        # print(torch.cat(f, dim=-1).shape)
        return torch.cat(f, dim=-1)  # [BatchSize, FeatureSize]


    def forward(self, x, y, require_grad=True):
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        pred = self.linear(torch.abs(feats0 - feats1))  # [N, dim]
        # pred = self.linear(feats0 - feats1)  # [N, dim]
        
        pred = torch.bmm(pred.unsqueeze(1), pred.unsqueeze(-1)).squeeze(-1)  # inner-prod
        # import pdb;pdb.set_trace()
        return torch.sqrt(pred - torch.sum(self.linear.bias**2)+1e-10).mean()  # [N, 1]
    
class VGG19(torch.nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()

        self.block1_conv1 = torch.nn.Conv2d(3, 64, (3,3), padding=(1,1), padding_mode='reflect')
        self.block1_conv2 = torch.nn.Conv2d(64, 64, (3,3), padding=(1,1), padding_mode='reflect')

        self.block2_conv1 = torch.nn.Conv2d(64, 128, (3,3), padding=(1,1), padding_mode='reflect')
        self.block2_conv2 = torch.nn.Conv2d(128, 128, (3,3), padding=(1,1), padding_mode='reflect')

        self.block3_conv1 = torch.nn.Conv2d(128, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv2 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv3 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv4 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')

        self.block4_conv1 = torch.nn.Conv2d(256, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv2 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv3 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv4 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsampling = torch.nn.AvgPool2d((2,2))

    def forward(self, image):
        
        # RGB to BGR
        image = image[:, [2,1,0], :, :]

        # [0, 1] --> [0, 255]
        image = 255 * image

        # remove average color
        image[:,0,:,:] -= 103.939
        image[:,1,:,:] -= 116.779
        image[:,2,:,:] -= 123.68

        # block1
        block1_conv1 = self.relu(self.block1_conv1(image))
        block1_conv2 = self.relu(self.block1_conv2(block1_conv1))
        block1_pool = self.downsampling(block1_conv2)

        # block2
        block2_conv1 = self.relu(self.block2_conv1(block1_pool))
        block2_conv2 = self.relu(self.block2_conv2(block2_conv1))
        block2_pool = self.downsampling(block2_conv2)

        # block3
        block3_conv1 = self.relu(self.block3_conv1(block2_pool))
        block3_conv2 = self.relu(self.block3_conv2(block3_conv1))
        block3_conv3 = self.relu(self.block3_conv3(block3_conv2))
        block3_conv4 = self.relu(self.block3_conv4(block3_conv3))
        block3_pool = self.downsampling(block3_conv4)

        # block4
        block4_conv1 = self.relu(self.block4_conv1(block3_pool))
        block4_conv2 = self.relu(self.block4_conv2(block4_conv1))
        block4_conv3 = self.relu(self.block4_conv3(block4_conv2))
        block4_conv4 = self.relu(self.block4_conv4(block4_conv3))

        return [block1_conv1, block1_conv2, block2_conv1, block2_conv2, block3_conv1, block3_conv2, block3_conv3, block3_conv4, block4_conv1, block4_conv2, block4_conv3, block4_conv4]

def slicing_loss(list_activations_generated, list_activations_example):
    
    loss = 0
    for l in range(len(list_activations_example)):
        # get dimensions
        b = list_activations_example[l].shape[0]
        dim = list_activations_example[l].shape[1]
        n = list_activations_example[l].shape[2]*list_activations_example[l].shape[3]
        # linearize layer activations and duplicate example activations according to scaling factor
        activations_example = list_activations_example[l].view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR)
        activations_generated = list_activations_generated[l].view(b, dim, n*SCALING_FACTOR*SCALING_FACTOR)
        # sample random directions
        Ndirection = dim
        device = activations_example.device  # Or any one of the tensors
        directions = torch.randn(Ndirection, dim).to(device)
        directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
        # project activations over random directions
        projected_activations_example = torch.einsum('bdn,md->bmn', activations_example, directions)
        projected_activations_generated = torch.einsum('bdn,md->bmn', activations_generated, directions)
        # sort the projections
        sorted_activations_example = torch.sort(projected_activations_example, dim=2)[0]
        sorted_activations_generated = torch.sort(projected_activations_generated, dim=2)[0]
        # L2 over sorted lists
        loss += torch.mean( (sorted_activations_example-sorted_activations_generated)**2 ) 
    return loss
vgg = VGG19()
vgg.load_state_dict(torch.load("./vgg19.pth"))
vgg.eval().to(device)
stsim_loss = STSIM_VGG([5900,10],grayscale=False).to(device).double()
def render_preview(filename, save_path):
    # Just copy the cached preview image from static
    src_path = os.path.join("static", "previews", f"{filename}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(src_path):
        Image.open(src_path).save(save_path)
    else:
        raise FileNotFoundError(f"Preview image not found: {src_path}")

def remap_strengths(signed_scalar):
    return np.exp(signed_scalar) if signed_scalar < 0 else signed_scalar
def power_normalize(tensor, gamma=0.5):
    return torch.sign(tensor) * (tensor.abs() ** gamma)
def CLIP_editing_gloss(generator, latent_s, weights_deltas, alpha, device):
    # s_dir = torch.load("dir_glossy_matte_clip.pt").to(device) * 0.5
    s_dir = torch.zeros_like(latent_s).to(device)
    s_dir[0,12308] = 1  # highlights
    # s_dir[0,12156] = 1  # contrast
    # s_dir[0,12172] = 1 # contrast + highlights
    # s_dir[0,12277] = 0.5  # brightness roughness
    # s_dir[0,12139] = 1   # showdows
    # s_dir[0,12175] = -1  # contrast
    # s_dir = power_normalize(s_dir,gamma=2)
    img_clip = generator.synthesis(ss=latent_s + s_dir * alpha, weights_deltas=weights_deltas, noise_mode='const').clamp(-1, 1)
    return (img_clip + 1) / 2

def CLIP_editing_rough(generator, latent_s, weights_deltas, alpha, device):
    # s_dir = torch.load("dir_rough_smooth.pt").to(device) * 5
    s_dir = torch.zeros_like(latent_s).to(device)
    s_dir[0,12155] = 1  # 
    img_clip = generator.synthesis(ss=latent_s + s_dir * alpha, weights_deltas=weights_deltas, noise_mode='const').clamp(-1, 1)
    return (img_clip + 1) / 2

def get_clip_similarity(img_tensor, feat, clip_model, device):
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = F.interpolate(img_tensor.to(device), size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor(OPENAI_DATASET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(OPENAI_DATASET_STD).view(1, 3, 1, 1).to(device)
    norm_tensor = (img_tensor - mean) / std
    img_feat = clip_model.encode_image(norm_tensor)
    img_feat = F.normalize(img_feat, dim=-1)
    return (img_feat @ feat.T).mean(dim=1)

def get_clip_feat(img_tensor, clip_model, device):
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = F.interpolate(img_tensor.to(device), size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor(OPENAI_DATASET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(OPENAI_DATASET_STD).view(1, 3, 1, 1).to(device)
    norm_tensor = (img_tensor - mean) / std
    img_feat = clip_model.encode_image(norm_tensor)
    img_feat = F.normalize(img_feat, dim=-1)
    return img_feat
def encode_prompts(prompt_list, tokenizer, clip_model, device):
    with torch.no_grad():
        tokens = tokenizer(prompt_list).to(device)
        features = clip_model.encode_text(tokens)
        return F.normalize(features, dim=-1)
def run_inference(filename, method, strength, pt_dir="real_latent"):
    pt_path = os.path.join(pt_dir, filename)
    data = torch.load(pt_path, map_location=device)

    # Detect format
    if isinstance(data, dict) and "s_code" in data:
        s_code = generator.synthesis.get_s_codes(data['s_code'].to(device)).to(device)
        weights_deltas = [w.to(device) if w is not None else None for w in data['delta_weights']]
    else:
        s_code = data.to(device)
        weights_deltas = None

    base_img = generator.synthesis(ss=s_code, weights_deltas=weights_deltas, noise_mode='const')
    img = (base_img.clamp(-1, 1) + 1) / 2
    img = img.squeeze(0)

    img_feat = get_clip_feat(img, clip_model, device)

    if method == "bs":
        edited = band_sifting_editing(img, effect="shine", strength=strength).to(device)
    elif method == "scurve":
        edited = S_Transformation(img, lam=remap_strengths(strength)).to(device)
    elif method == "clip":
        edited = CLIP_editing_gloss(generator, s_code, weights_deltas, alpha=strength, device=device)
        edited = edited.squeeze(0)
    else:
        edited = img

    save_path = f"static/tmp/{filename}_{method}_{strength}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(edited, save_path)

    with open("prompts.json", "r") as f:
        prompts = json.load(f)

    feat_pos = encode_prompts(prompts["glossy"], tokenizer, clip_model, device)
    feat_neg = encode_prompts(prompts["matte"], tokenizer, clip_model, device)
    sim_glossy = get_clip_similarity(edited, feat_pos, clip_model, device).item()
    sim_matte = get_clip_similarity(edited, feat_neg, clip_model, device).item()

    sim_img = get_clip_similarity(edited, img_feat, clip_model, device).item()
    stsim = stsim_loss(img.unsqueeze(0).double(), edited.unsqueeze(0).double()).item()
    sw = slicing_loss(vgg(img.unsqueeze(0)),vgg(edited.unsqueeze(0))).item()

    return f"/{save_path}", sim_glossy, sim_matte, sim_img,stsim,sw


def run_inference_roughness(filename, method, strength, pt_dir="real_latent"):
    pt_path = os.path.join(pt_dir, filename)
    data = torch.load(pt_path, map_location=device)

    # Detect format
    if isinstance(data, dict) and "s_code" in data:
        s_code = generator.synthesis.get_s_codes(data['s_code'].to(device)).to(device)
        weights_deltas = [w.to(device) if w is not None else None for w in data['delta_weights']]
    else:
        s_code = data.to(device)
        weights_deltas = None

    base_img = generator.synthesis(ss=s_code, weights_deltas=weights_deltas, noise_mode='const')
    img = (base_img.clamp(-1, 1) + 1) / 2
    img = img.squeeze(0)
    img_feat = get_clip_feat(img, clip_model, device)

    if method == "bs":
        edited = band_sifting_editing(img, effect="rough", strength=strength).to(device)
    elif method == "clip":
        edited = CLIP_editing_rough(generator, s_code, weights_deltas, alpha=strength, device=device)
        edited = edited.squeeze(0)
    else:
        edited = img

    save_path = f"static/tmp/{filename}_rough_{method}_{strength}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(edited, save_path)

    with open("prompts.json", "r") as f:
        prompts = json.load(f)

    feat_pos = encode_prompts(prompts["rough"], tokenizer, clip_model, device)
    feat_neg = encode_prompts(prompts["smooth"], tokenizer, clip_model, device)
    sim_rough = get_clip_similarity(edited, feat_pos, clip_model, device).item()
    sim_smooth = get_clip_similarity(edited, feat_neg, clip_model, device).item()
    sim_img = get_clip_similarity(edited, img_feat, clip_model, device).item()
    stsim = stsim_loss(img.unsqueeze(0).double(), edited.unsqueeze(0).double()).item()
    sw = slicing_loss(vgg(img.unsqueeze(0)),vgg(edited.unsqueeze(0))).item()

    return f"/{save_path}", sim_rough, sim_smooth, sim_img,stsim,sw

