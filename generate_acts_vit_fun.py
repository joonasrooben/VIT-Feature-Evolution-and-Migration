import torch
import sys
import argparse, os
from openood.networks import ViT_B_16

sys.path.append('..')
sys.path.append('../openood/OpenOOD/')
import numpy as np
#import importlib
#import openood
import torch.nn as nn
from openood.datasets import get_dataloader,get_ood_dataloader
import yaml 
#import openood.datasets
from openood.utils.config import Config
from torch.nn.utils import spectral_norm
from pathlib import Path

def patches_hidden_states(model, x):
    """
    Returns a list of [B, D] tensors: CLS embedding after each encoder block.
    """
    model.eval()
    hs = []
    hooks = []

    # Each block output is [B, seq_len, D]; patch tokens are at index 1:
    def make_hook():
        def hook(module, inp, out):
            hs.append(out[:, 1:].detach())
        return hook

    # torchvision ViT blocks live here:
    # model.encoder.layers is a Sequential of EncoderBlock's
    for blk in model.encoder.layers:
        hooks.append(blk.register_forward_hook(make_hook()))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return hs  # length = num_layers, each element shape [B, D]

def cls_hidden_states(model, x):
    """
    Returns a list of [B, D] tensors: CLS embedding after each encoder block.
    """
    model.eval()
    hs = []
    hooks = []

    # Each block output is [B, seq_len, D]; CLS token is at index 0
    def make_hook():
        def hook(module, inp, out):
            hs.append(out[:, 0].detach())
        return hook

    # torchvision ViT blocks live here:
    # model.encoder.layers is a Sequential of EncoderBlock's
    for blk in model.encoder.layers:
        hooks.append(blk.register_forward_hook(make_hook()))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return hs  # length = num_layers, each element shape [B, D]

def get_model_outputs(data_loader, model, layer_num=-2, logits_pos = -1, return_name = False, patches_on = False, train = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_all, logits_all, labels, classes, patches = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if return_name:
                labels.extend(batch["image_name"])
            classes.append(batch["label"].long().to(device))
            cls_batch = cls_hidden_states(model,batch["data"].float().to(device))
            if patches_on:
                if train:
                    perm = torch.randperm(batch["data"].size(0))
                    k = int(batch["data"].size(0)*0.05)
                    idx = perm[:k]
                    samples = batch["data"][idx]
                else:
                    samples = batch["data"]

                patches_batch = patches_hidden_states(model,samples.float().to(device))
                patches_layers = torch.stack(patches_batch, dim=1)  # (B, L, pxp, 768)
                patches.append(patches_layers)
            
            cls_layers = torch.stack(cls_batch, dim=1)  # (B, L, 768)
            cls_all.append(cls_layers)
            logits_all.append(model(batch["data"].float().to(device)))

#    labels_tensor = torch.cat(labels, dim=0)
    classes_tensor = torch.cat(classes, dim = 0)
    cls_tensor   = torch.cat(cls_all, dim=0)      # (N, L, 768)
    logits_tensor = torch.cat(logits_all, dim=0)   # (N, n_labels)



    output = {
        "logit": logits_tensor,
        #"softmax": torch.nn.functional.softmax(torch.stack(logits), dim=1),
        "layer_x": cls_tensor,
        "classes": classes_tensor,
    }
    if patches_on:
        output["patches"] = torch.cat(patches, dim=0) #(N,L,pxp,768)
    if return_name:
        output["names"] = labels
    return output   



def generate_acts_vit(args):

    openood_path = Path("../openood/OpenOOD")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ind_name = args["ind_name"] #cif10, cif100,imagenet200
    num_classes = args["num_classes"]
    
    # load the model

    if args["pretrained_vit_IN1K"]:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        net = vit_b_16( weights = ViT_B_16_Weights.IMAGENET1K_V1, progress = True) 
        net.cuda()
        net.eval()
        with open(openood_path / f'results/{args["ind_name"]}_vit-b-16/s{args["seed"]}/config_gen.yml') as stream:
            try:
                conf = Config(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)


    else:
        net = ViT_B_16(args["img_size"], num_classes=num_classes , hidden_dim=args["vit_size"]//4 , mlp_dim = args["vit_size"], num_heads=args["num_heads"])

        if args["step"] != 0:
            net.load_state_dict(
                torch.load(openood_path / f'results/{args["ind_name"]}_vit-b-16_base_e{args["total_epochs"]}_lr0.1_default/s{args["seed"]}/model_epoch{args["epoch"]}_step{args["step"]}.ckpt', map_location=device)
            )
            net.cuda()
            net.eval()
        else:
            net.load_state_dict(
                torch.load(openood_path / f'results/{args["ind_name"]}_vit-b-16_base_e{args["total_epochs"]}_lr0.1_default/s{args["seed"]}/model_epoch{args["epoch"]}.ckpt', map_location=device)
            )
            net.cuda()
            net.eval()

        with open(openood_path / f'results/{args["ind_name"]}_vit-b-16_base_e{args["total_epochs"]}_lr0.1_default/s{args["seed"]}/config_gen.yml') as stream:
            try:
                conf = Config(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    loader_dict = get_dataloader(conf)
    #ood_loader_dict = get_ood_dataloader(conf)
    #ood_loader_far , ood_loader_near = ood_loader_dict["farood"], ood_loader_dict["nearood"]

    ind_dict = list(loader_dict.keys())
    #print(ind_dict)
    #near_ood_dict = list(ood_loader_near.keys())
    #far_ood_dict = list(ood_loader_far.keys())

    #path = "./cif10_acts_s2"
    path = openood_path / f'vit_{args["ind_name"]}_s{args["seed"]}/epoch{args["epoch"]}_step{args["step"]}'
    if not os.path.exists(path):
        # Create the folder (including any necessary parent directories)
        os.makedirs(path)
        print(f"Folder created: {path}")
    else:
        print(f"Folder already exists: {path}")

    # ---------- save ---------- #
    if ind_name in ["shapes3d", "imagenet_mixed10_balanced"]:
        return_name = True
    else:
        return_name = False

    for key in ind_dict:
        print(key)
        if args["train_patches"] and (key == "train"):
            buf_data = get_model_outputs(loader_dict[key],net,-1,-2,return_name = return_name, patches_on = True, train = True)
            np.save(f"{path}/vit_{ind_name}_{key}_patches.npy",buf_data["patches"].detach().cpu().numpy())
        if args["test_patches"] and (key == "test"):
            buf_data = get_model_outputs(loader_dict[key],net,-1,-2,return_name = return_name, patches_on = True)
            np.save(f"{path}/vit_{ind_name}_{key}_patches.npy",buf_data["patches"].detach().cpu().numpy())
        else:
            buf_data = get_model_outputs(loader_dict[key],net,-1,-2,return_name = return_name)
        np.save(f"{path}/vit_{ind_name}_{key}.npy",buf_data["layer_x"].detach().cpu().numpy())
        np.save(f"{path}/vit_{ind_name}_{key}_y.npy",buf_data["classes"].detach().cpu().numpy())
        np.save(f"{path}/vit_{ind_name}_{key}_logits.npy",buf_data["logit"].detach().cpu().numpy())
        if return_name:
            np.save(f"{path}/vit_{ind_name}_{key}_names.npy",np.array(buf_data["names"]))
