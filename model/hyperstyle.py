import math
import torch
from torch import nn
import copy
from argparse import Namespace

from model.hypernetworks.hypernetwork import SharedWeightsHyperNetResNetG3
# from train_latent import VGG19, slicing_loss
from torchvision.utils import save_image
import lpips
import os
import tqdm
SCALING_FACTOR = 1
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
    
    # generate VGG19 activations
    # list_activations_generated = vgg(image_generated)
    # list_activations_example   = vgg(image_example)
    
    # iterate over layers
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

class HyperStyle(nn.Module):

    def __init__(self, opts):
        super(HyperStyle, self).__init__()
        self.set_opts(opts)
        self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.hypernet = self.set_hypernet()
        # ckpt_path = "/data/zxq0322/CLIP-Texture-Attribute-Editing/hypernet_output/weights/hypernet_epoch_001.pt"
        # self.hypernet.load_state_dict(torch.load(ckpt_path, map_location=self.opts.device))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.vgg = VGG19()
        self.vgg.load_state_dict(torch.load("/data/zxq0322/stylegan3-texture-analysis-synthesis/vgg19.pth"))
        self.vgg.eval().to(self.opts.device)
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.opts.device)

    def load_enc_dec(self,encoder, G):
        self.w_encoder=encoder.eval().to(self.opts.device)
        self.decoder = G.synthesis.to(self.opts.device)

    def set_hypernet(self):
        networks = SharedWeightsHyperNetResNetG3(opts=self.opts).to(self.opts.device)
        return networks

    def forward_once(self, x, resize=False, input_code=False, return_latents=False,
                return_weight_deltas_and_codes=False, weights_deltas=None, y_hat=None,
                codes=None):

        if input_code:
            codes = x
        else:
            # concatenate original input with w-reconstruction or current reconstruction
            x_input = torch.cat([x, y_hat], dim=1)
            # pass through hypernet to get per-layer deltas
            hypernet_outputs = self.hypernet(x_input)
            if weights_deltas is None:
                weights_deltas = hypernet_outputs
            else:
                weights_deltas = [weights_deltas[i] + hypernet_outputs[i] if weights_deltas[i] is not None else None
                                  for i in range(len(hypernet_outputs))]

        images = self.decoder(ws=codes, weights_deltas=weights_deltas)

        if resize:
            images = self.face_pool(images)

        if return_latents and return_weight_deltas_and_codes:
            return images, codes, weights_deltas, codes, y_hat
        elif return_latents:
            return images, codes
        elif return_weight_deltas_and_codes:
            return images, weights_deltas, codes
        else:
            return images
    def forward(self, x, resize=False, input_code=False, return_latents=False,
            return_weight_deltas_and_codes=True, weights_deltas=None, y_hat=None, codes=None,
            lr=1e-2, epoches=300, print_freq=25):
    
        self.hypernet.eval()  # freeze hypernet
        images = []

        if input_code:
            codes = x
        else:
            if y_hat is None:
                assert self.opts.load_w_encoder
                y_hat, codes = self.optimize_latent_codes(x)

                img_out = (y_hat.clone().detach().cpu().clamp(-1, 1) + 1) / 2
                images.append(img_out)

        # Get initial delta_weights from hypernet (frozen)
        with torch.no_grad():
            x_input = torch.cat([x, y_hat], dim=1)
            weights_deltas = self.hypernet(x_input)
        
        # Now treat delta_weights as learnable
        for i in range(len(weights_deltas)):
            if weights_deltas[i] is not None:
                weights_deltas[i].requires_grad = True

        # Filter out None entries from weights_deltas
        trainable_deltas = [w for w in weights_deltas if w is not None]
        optimizer = torch.optim.Adam(trainable_deltas, lr=lr)


        for epoch in range(epoches):
            optimizer.zero_grad()

            y_hat = self.decoder(ws=codes, weights_deltas=weights_deltas)
            act_fake = self.vgg(y_hat)
            act_real = self.vgg(x)
            loss = slicing_loss(act_fake, act_real)
            loss.backward()
            optimizer.step()

            if epoch > 0 and (epoch % print_freq == 0 or epoch == epoches - 1):
                img_out = (y_hat.clone().detach().cpu().clamp(-1, 1) + 1) / 2
                images.append(img_out)

        return images, weights_deltas, codes
    
    def set_opts(self, opts):
        self.opts = opts

    def __get_initial_inversion(self, x, resize=True):
        # get initial inversion and reconstruction of batch
        with torch.no_grad():
            return self.__get_w_inversion(x, resize)

    def __get_w_inversion(self, x, resize=True):
        if self.w_encoder.training:
            self.w_encoder.eval()
        with torch.no_grad():
            codes = self.w_encoder(x).mean(dim=1)
            codes = codes.unsqueeze(1).repeat(1, 16 ,1)
        y_hat = self.decoder(ws=codes,
                                weights_deltas=None)
        return y_hat, codes
    def optimize_latent_codes(self, x, steps=400, lr=0.001):
        """Optimize latent codes (ws) with fixed encoder and decoder."""
        with torch.no_grad():
            codes = self.w_encoder(x).mean(dim=1)
            # print(codes.device)
            codes = codes.unsqueeze(1).repeat(1, 16 ,1)
        mse_loss = nn.MSELoss()
        
        codes = codes.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([codes], lr=lr)
        sw_losses = []

        for step in range(steps):
            y_hat = self.decoder(ws=codes, weights_deltas=None)
            # print(y_hat.device)
            act_fake = self.vgg(y_hat)
            act_real = self.vgg(x.to(self.opts.device))
            loss = slicing_loss(act_fake, act_real)
            for f, r in zip(act_fake, act_real):
                loss += 0.005 * mse_loss(f, r)
            # loss = self.loss_fn_vgg(x, y_hat)
            # print(loss.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sw_losses.append(loss.item())
            # print(f"[Latent Opt] Step {step+1}/{steps}, Loss: {loss.item():.4f}")

        return y_hat.detach(), codes.detach() 

    def train(self, dataloader, lr=1e-4, epochs=10, print_freq=100, save_dir="./train_output"):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'weights'), exist_ok=True)

        self.hypernet.train()
        self.w_encoder.eval()
        self.decoder.eval()
        self.vgg.eval()

        optimizer = torch.optim.Adam(self.hypernet.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        step = 0
        for epoch in range(epochs):
            pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            for batch_idx, batch in enumerate(pbar):
                x = batch.to(self.opts.device)

                # Initial inversion
                y_hat_init, codes = self.optimize_latent_codes(x, steps=100)

                # Forward pass with hypernetwork
                y_hat, weights_deltas, _ = self.forward_once(
                    x=x,
                    resize=False,
                    input_code=False,
                    return_latents=False,
                    return_weight_deltas_and_codes=True,
                    weights_deltas=None,
                    y_hat=y_hat_init,
                    codes=codes
                )

                # Compute loss
                act_fake = self.vgg(y_hat)
                act_real = self.vgg(x)
                loss = slicing_loss(act_fake, act_real)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update tqdm progress bar
                pbar.set_postfix({"SW Loss": f"{loss.item():.4f}"})

                # Optionally save output
                if step % print_freq == 0:
                    with torch.no_grad():
                        # Normalize both to [0, 1]
                        x_vis = (x.clone().detach().cpu().clamp(-1, 1) + 1) / 2
                        y_vis = (y_hat.clone().detach().cpu().clamp(-1, 1) + 1) / 2

                        # Concatenate along width (dim=3)
                        comparison = torch.cat([x_vis, y_vis], dim=3)  # B x C x H x (2*W)

                        save_path = os.path.join(save_dir, f"step_{step:05d}.png")
                        save_image(comparison, save_path)

                step += 1
            # scheduler.step()
            torch.save(
                self.hypernet.state_dict(),
                os.path.join(save_dir, f'weights/hypernet_epoch_{epoch+1:03d}.pt')
            )
