from __future__ import annotations
import os
os.environ["SCIPY_ARRAY_API"] = "1"
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from tqdm import tqdm

from src.base_sae import BatchTopKSparseAutoencoder, SparseAutoencoder, TopKSparseAutoencoder
from src.sae_utils import  load_checkpoint, save_checkpoint, train_sae, get_saes_similarities, concept_summary_stats, plot_metrics_figure, linear_cka, train_sae_fast
from src.utils import load_all_data,  _ensure_dir, _device_or_default
from src.plotting import _plot_heatmap, _plot_two_heatmaps_with_acc_and_images, plot_top_activating_images_per_class
from src.probing import plot_subclass_probe_heatmap_cached, precompute_shared_subclass_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _run_seed(base_seed: int, run_idx: int, epoch: int, layer: int) -> int:
    return int(base_seed + run_idx * 100000 + epoch * 100 + layer)


def _build_sae(input_dim: int, args: Dict[str, Any]) -> SparseAutoencoder: ##proble?
    code_dim = int(input_dim * args["expansion_coef"])
    sae_type = args["sae_type"]
    if sae_type == "batch_top_k_sae":
        return BatchTopKSparseAutoencoder(
            input_dim=input_dim,
            code_dim=code_dim,
            tied_weights=args["tied"],
            use_circuits_implementation=args["circuits"],
            k=args["k"],
        )
    if sae_type == "top_k_sae":
        return TopKSparseAutoencoder(
            input_dim=input_dim,
            code_dim=code_dim,
            tied_weights=args["tied"],
            use_circuits_implementation=args["circuits"],
            k=args["k"],
        )
    return SparseAutoencoder(
        input_dim=input_dim,
        code_dim=code_dim,
        tied_weights=args["tied"],
        use_circuits_implementation=args["circuits"],
    )


def _checkpoint_path(args: Dict[str, Any], epoch: int, layer: int, run_idx: int) -> Path:
    folder = Path(args["checkpoint_folder"]).parent 
    _ensure_dir(folder)
    k_tag = f"_k{args['k']}" if "k" in args else ""
    exp_tag = f"_exp{args['expansion_coef']}"
    special_token = f'_trained_{args["trained_on"]}' if args["trained_on"] != None else ""
    name = (
        f"{args['backbone']}_{args['sae_type']}_run{run_idx}_seed{args['seed']}"
        f"_epoch{epoch}_step{args['step']}_layer{layer}"
        f"_reg{args['norm']}_tied{args['tied']}_epochs{args['sae_epochs']}{k_tag}{exp_tag}{special_token}.pt"
    )
    return folder / name


def _load_all_data_cached(
    epoch: int,
    layer: int,
    args: Dict[str, Any],
    cache: Optional[Dict[Tuple[int, int], Dict[str, Any]]],
    eval_mode = False
) -> Dict[str, Any]:
    key = (epoch, layer, bool(eval_mode), args["eval_split"])
    if cache is not None and key in cache:
        return cache[key]
    if args["trained_on"] in ["mix", "patches"]:
        train_patches = True
    else: 
        train_patches = False

    if eval_mode:
        train_patches = False
    try:
        data = load_all_data(
            epoch,
            args["step"],
            args["seed"],
            args["norm"],
            args["keywords"],
            layer_nr=layer,
            backbone=args["backbone"],
            ind_name = args["ind_name"], 
            train_patches =train_patches,
            test_patches = args["test_patches"],
            eval_mode = eval_mode,
            eval_split = args["eval_split"],
            activation_root=args.get("activation_root", "../openood/OpenOOD"),

        )
    except FileNotFoundError:
        if "activation_root" in args:
            raise  # Explicit activation stores must already have been exported.
        print("Gathering activations")
        args_dict_2 = {}
        args_dict_2["ind_name"] = args["ind_name"]
        args_dict_2["epoch"] = epoch
        args_dict_2["step"] = args["step"]
        args_dict_2["num_classes"] = args["num_classes"]
        args_dict_2["seed"] = args["seed"].split("s")[-1]
        args_dict_2["total_epochs"] = args["total_vit_epochs"]
        args_dict_2["img_size"] = args["img_size"]
        args_dict_2["vit_size"] = args["vit_size"]
        args_dict_2["num_heads"] = args["num_heads"]
        args_dict_2["pretrained_vit_IN1K"] = args["pretrained_vit_IN1K"]
        args_dict_2["train_patches"] = train_patches
        args_dict_2["test_patches"] = args["test_patches"]
        if args["backbone"] == "vit_mae":
            from generate_acts_vit_mae import generate_acts_vit_mae
            generate_acts_vit_mae(args_dict_2)
        else:
            if args["seed"] in ['_s7', '_s8']:
                from generate_acts_vit_fun_tiny import generate_acts_vit
                args_dict_2["train_imglist"] = "/gpfs/space/projects/mlgroup/data/benchmark_imglist/imagenet/train_imagenet100k.txt" #"/gpfs/space/projects/mlgroup/data/benchmark_imglist/imagenet200/train_mixed10_balanced.txt"
                #args_dict_2["val_imglist"] = "/gpfs/space/projects/mlgroup/data/benchmark_imglist/imagenet200/val_mixed10_balanced.txt"
                args_dict_2["test_imglist"] = "/gpfs/space/projects/mlgroup/data/benchmark_imglist/imagenet/test_imagenet.txt"
                generate_acts_vit(args_dict_2)
            else:
                from generate_acts_vit_fun import generate_acts_vit
                generate_acts_vit(args_dict_2)


        data = load_all_data(
            epoch,
            args["step"],
            args["seed"],
            args["norm"],
            args["keywords"],
            layer_nr=layer,
            backbone=args["backbone"],
            ind_name = args["ind_name"],
            train_patches = train_patches,
            test_patches = args["test_patches"],
            eval_mode=eval_mode,
            eval_split=args["eval_split"],
            activation_root=args.get("activation_root", "../openood/OpenOOD"),

        )

    if cache is not None:
        cache[key] = data
    return data

def _load_sae_for_finetune(path: Path, device: str) -> SparseAutoencoder:
    sae = load_checkpoint(path, device=device)
    sae.to(device)
    return sae

def _load_sae_for_eval(path: Path, device: str) -> SparseAutoencoder:
    sae = load_checkpoint(path, device=device)
    sae.to(device)
    sae.eval()
    return sae


def _finetune_sae(
    args: Dict[str, Any],
    ref_ckpt,
    ref_epoch,
    ref_layer,
    cache ,
):
    train_device = _device_or_default(args.get("train_device"))
    sae = _load_sae_for_finetune(ref_ckpt, train_device)
    data = _load_all_data_cached(ref_epoch, ref_layer, args, cache)
    if args["trained_on"] in ["cls", "mix", None]:
        latents = torch.from_numpy(data[args["train_split"]]["activations"]).float()
    if args["trained_on"] in ["mix","patches"]:
        patch_latents = torch.from_numpy(data[args["train_split"]]["patches"]).float().reshape((-1,args["vit_size"]//4))
        if args["trained_on"] == "patches":
            latents = patch_latents
        else:
            latents = torch.cat([latents, patch_latents],dim = 0)

    train_sae_fast(
        sae,
        latents,
        epochs=5, ##nr of epochs to finetune
        batch_size=args["batch_size"],
        lr=args["lr"],
        l1_lambda=args["l1_lambda"],
        z_l2_lambda=args["l2_lambda"],
        verbose_every=args["verbose_every"],
        device=train_device,
    )
    latents_np = data[args["eval_split"]]["activations"]
    latent = torch.from_numpy(latents_np).to(train_device)
    sae.eval()
    with torch.no_grad():
        acts, recon = sae.encode(latent,only_z=False)
        criterion = torch.nn.MSELoss(reduction="mean")
        recon_loss = criterion(recon, latent)
    return acts.cpu(), recon_loss.item(), sae

def _train_all_saes(
    args, epochs, layers, ref_epoch, ref_layer, cache,
):
    return train_sae_grid(
        args, epochs, layers, ref_epoch, ref_layer,
        load_data=lambda epoch, layer: _load_all_data_cached(
            epoch, layer, args, cache, eval_mode=False,
        ),
        checkpoint_path=lambda epoch, layer, run_idx: _checkpoint_path(
            args, epoch, layer, run_idx,
        ),
    )

def train_sae_grid(
    args, epochs, layers, ref_epoch, ref_layer,
    *, load_data, checkpoint_path, trainer=None,):
    device = str(_device_or_default(args.get("train_device")))
    mode = args["trained_on"] or "cls"
    if mode not in {"cls", "patches", "mix"}:
        raise ValueError(f"Unknown trained_on: {mode}")

    pairs = ([(ref_epoch, ref_layer)] if args["fixed_sae"]
             else [(epoch, layer) for epoch in epochs for layer in layers])
    paths = {}
    
    for run_idx in range(args["sae_runs"]):
        # Preserve the script's common initialization seed within each run.
        seed = _run_seed(
            args["run_seed_base"], run_idx, ref_epoch, ref_layer
        )

        for epoch, layer in pairs:
            path = Path(checkpoint_path(epoch, layer, run_idx))
            paths[(run_idx, epoch, layer)] = path
            if args["skip_existing"] and path.exists():
                continue

            split = load_data(epoch, layer)[args["train_split"]]
            parts = []

            if mode in {"cls", "mix"}:
                parts.append(
                    torch.as_tensor(split["activations"]).float()
                )

            if mode in {"patches", "mix"}:
                if split.get("patches") is None:
                    raise ValueError(
                        "Patch training requested, but patches are missing."
                    )
                patches = torch.as_tensor(split["patches"]).float()
                parts.append(patches.reshape(-1, patches.shape[-1]))

            latents = torch.cat(parts, dim=0)
            if latents.ndim != 2 or min(latents.shape) == 0:
                raise ValueError(
                    "Training activations must have nonempty shape [N, D]."
                )

            seed_everything(seed)
            sae = _build_sae(latents.shape[1], args)
            sae.renorm_decoder_()

            fit = trainer
            if fit is None:
                fit = (train_sae
                        if (epoch, layer) == (ref_epoch, ref_layer)
                        else train_sae_fast
                    )

            fit(
                sae, latents,
                epochs=args["sae_epochs"],
                batch_size=args["batch_size"],
                lr=args["lr"],
                l1_lambda=args["l1_lambda"],
                z_l2_lambda=args["l2_lambda"],
                verbose_every=args["verbose_every"],
                device=device,
            )

            path.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                key: args[key]
                for key in (
                    "seed", "step", "backbone",
                    "trained_on", "expansion_coef",
                )
            }
            meta.update(
                epoch=epoch, layer=layer,
                run=run_idx, run_seed=seed,
            )
            save_checkpoint(sae, path, meta=meta)

    return paths

## broken topk return 
import time
def _evaluate_run_combined(
    run_idx: int,
    args: Dict[str, Any],
    epochs: List[int],
    layers: List[int],
    ref_epoch: int,
    ref_layer: int,
    cache: Optional[Dict[Tuple[int, int], Dict[str, Any]]],
) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    eval_device = _device_or_default(args.get("eval_device"))
    start = time.time()
    criterion = torch.nn.MSELoss(reduction="mean")
    ref_ckpt = _checkpoint_path(args, ref_epoch, ref_layer, run_idx)
    if not ref_ckpt.exists():
        raise FileNotFoundError(f"Missing reference SAE checkpoint: {ref_ckpt}")

    ref_sae = _load_sae_for_eval(ref_ckpt, eval_device)

    ref_data = _load_all_data_cached(ref_epoch, ref_layer, args, cache)
    ref_latents = torch.from_numpy(ref_data[args["eval_split"]]["activations"]).float().to(eval_device)
    ref_labels = torch.from_numpy(ref_data[args["eval_split"]]["labels"])
    del ref_data 

    with torch.no_grad():
        ref_sae_latents = ref_sae.encode(ref_latents).cpu()

    results_dir = Path(args["results_dir"]).parent.parent / "evo_patterns"
    _ensure_dir(results_dir)
    print(results_dir)

    total_nr_concepts = ref_sae_latents.shape[1]
    
    cor_res = np.full((total_nr_concepts ,len(layers), len(epochs)), np.nan, dtype=float)
    top_res = np.full((total_nr_concepts ,len(layers), len(epochs)), np.nan, dtype=float)

    pre_losses = np.full((len(layers), len(epochs)), np.nan, dtype=float)
    ckas = np.full((len(layers), len(epochs)), np.nan, dtype=float)
            
    plots_dir = Path(args["plots_dir"]) / f"run_{run_idx}" /f"combined_{args['tag']}"
    _ensure_dir(plots_dir)

    concepts_stats = concept_summary_stats(ref_sae_latents, ref_labels, top_k=None)

    stats_np = []
    for key in concepts_stats:
        c = concepts_stats[key]
        stats_np.append([c["sparsity"],c["mean_activation"],c["label_entropy"]])

    plot_metrics_figure(concepts_stats, save_path = plots_dir / f"concept_stats_run_{run_idx}_ref_e{ref_epoch}_l{ref_layer}.png")
    end = time.time()
    print("pre time", end-start)
    for li, layer in enumerate(layers):
        print("Layer:", layer)
        for ei, epoch in tqdm(enumerate(epochs),desc=f"layer {layer}"):
    
            data = _load_all_data_cached(epoch, layer, args, cache,eval_mode = True)
            pre_loss = None 
            ckpt_latents = torch.from_numpy(data).float().to(eval_device)
            
            cka = linear_cka(ref_latents, ckpt_latents)
            ckas[li,ei] = cka
 
            #independent SAE
            ckpt = _checkpoint_path(args, epoch, layer, run_idx)
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing SAE checkpoint: {ckpt}")
            sae = _load_sae_for_eval(ckpt, eval_device)
            with torch.no_grad():
                acts_a = sae.encode(ckpt_latents)
                    
            sim_res_dict = get_saes_similarities(
                acts_a.to(eval_device),
                ref_sae_latents.to(eval_device),
                False,
                return_cos = False,
                method=args["align_method"],
                min_sim=args["align_min_sim"],
                compute_stability=args["align_compute_stability"],
                compute_chance=args["align_compute_chance"],
                device=eval_device,
                return_similarity = True,
                k=args["topk_k"],
                metric=args["topk_metric"],
                threshold=args["topk_threshold"],
                act_threshold=args["topk_act_threshold"]
            )
            cor_res[:,li,ei] = sim_res_dict["cor_sim"].max(0) 
            top_res[:,li,ei] = sim_res_dict["topk_sim"].max(0)
            del sim_res_dict, acts_a

            ## FIXED SAE
            with torch.no_grad():
                acts_a = ref_sae.encode(ckpt_latents)  # Tracking readouts
                recon, _ = ref_sae(ckpt_latents)  # Reconstruction
            recon_loss = criterion(recon, ckpt_latents)
            pre_loss = recon_loss.item()

            sim_res_dict = get_saes_similarities(
                acts_a.to(eval_device),
                ref_sae_latents.to(eval_device),
                False,
                return_cos = False,
                method=args["align_method"],
                min_sim=args["align_min_sim"],
                compute_stability=args["align_compute_stability"],
                compute_chance=args["align_compute_chance"],
                device=eval_device,
                return_similarity = True,
                k=args["topk_k"],
                metric=args["topk_metric"],
                threshold=args["topk_threshold"],
                act_threshold=args["topk_act_threshold"]
            )
            cor_res[:,li,ei] = np.maximum( cor_res[:,li,ei], sim_res_dict["cor_sim"].diagonal(0))
            top_res[:,li,ei] = np.maximum( top_res[:,li,ei], sim_res_dict["topk_sim"].diagonal(0))
            pre_losses[li,ei] = pre_loss

            del sim_res_dict, data, ckpt_latents
            torch.cuda.empty_cache()
   
    np.savez(results_dir / f"evolution_patterns_e{ref_epoch}_l{ref_layer}.npz", cor_res = cor_res, top_res =top_res, concepts = np.array(stats_np),epochs = epochs, cka = ckas, pre_loss = pre_losses)
    return None

def _evaluate_loss(
    run_idx: int,
    args: Dict[str, Any],
    epochs: List[int],
    layers: List[int],
    cache: Optional[Dict[Tuple[int, int], Dict[str, Any]]],
) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    eval_device = _device_or_default(args.get("eval_device"))
    criterion = torch.nn.MSELoss(reduction="mean")

    results_dir = Path(args["results_dir"]).parent.parent / "evo_patterns" / "loss"
    _ensure_dir(results_dir)
    print(results_dir)

    pre_losses = np.full((len(layers), len(epochs)), np.nan, dtype=float)
    
    for li, layer in enumerate(layers):
        print("Layer:", layer)
        for ei, epoch in tqdm(enumerate(epochs),desc=f"layer {layer}"):
    
            data = _load_all_data_cached(epoch, layer, args, cache,eval_mode = True)
            ckpt_latents = torch.from_numpy(data).float().to(eval_device)
            
            ckpt = _checkpoint_path(args, epoch, layer, run_idx)
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing SAE checkpoint: {ckpt}")
            sae = _load_sae_for_eval(ckpt, eval_device)
            with torch.no_grad():
                _ ,recon = sae.encode(ckpt_latents, only_z= False)
            
            recon_loss = criterion(recon, ckpt_latents)
            pre_loss = recon_loss.item()
            pre_losses[li,ei] = pre_loss

    np.savez(results_dir / f"pre_losses.npz", pre_losses = pre_losses)
    return None

def _save_concept_evo_pattern(
    run_idx: int,
    args: Dict[str, Any],
    epochs: List[int],
    layers: List[int],
    ref_epoch: int,
    ref_layer: int,
    method : str = "max",
    want_images : bool = True,
    eval_device = None
) -> None:

    results_dir = Path(args["results_dir"]).parent.parent / "evo_patterns"
    evo_pat_path = results_dir / f"evolution_patterns_e{ref_epoch}_l{ref_layer}.npz"
    evo_patterns = np.load(evo_pat_path)
    total_nr_concepts = args["vit_size"]//4*args["expansion_coef"]
    
    cor_res = evo_patterns["cor_res"]
    top_res = evo_patterns["top_res"]
    try:
        pre_losses = evo_patterns["pre_loss"]
        ckas = evo_patterns["cka"]
    except:
        print("no cka, no loss")

    concepts_stats = evo_patterns["concepts"]
    epochs = evo_patterns["epochs"]

    plots_dir = Path(args["plots_dir"]) / f"run_{run_idx}" /f"combined_{args['tag']}"
    _ensure_dir(plots_dir)

    accs = []
    if not args["pretrained_vit_IN1K"]:
        if args.get("log_path") is not None or args["num_heads"] == 3:
            log_path = Path(args.get("log_path", f'../openood/OpenOOD/results/{args["ind_name"]}_vit-b-16_base_e{args["total_vit_epochs"]}_lr0.1_default/s{args["seed"].split("s")[-1]}/log.txt'))
            with log_path.open("r") as f:
                for line in f:
                    if "Val Acc" in line:
                        accs.append(float(line.split("Acc")[1].strip()))
        else:
            if args["seed"] == "_s8":
                log_path = Path(f'../openood/OpenOOD/results/vit_tiny_ffcv/log.out')
            elif args["seed"] == "_s7":
                log_path = Path(f'../openood/OpenOOD/results/vit_tiny_a1_mixed/log.out')
            with log_path.open("r") as f:
                for line in f:
                    if "val_acc" in line:
                        accs.append(float(line.split("val_acc1=")[1].split(" ")[0].strip()))
        accs = np.array(accs)[epochs]
    else:
        accs = np.array([81])


    ref_ckpt = _checkpoint_path(args, ref_epoch, ref_layer, run_idx)
    if not ref_ckpt.exists():
        raise FileNotFoundError(f"Missing reference SAE checkpoint: {ref_ckpt}")
    ref_sae = _load_sae_for_eval(ref_ckpt, eval_device)
    cache = {} if args["cache_activations"] else None
    ref_data = _load_all_data_cached(ref_epoch, ref_layer, args, cache, eval_mode=True)
    ref_latents = torch.from_numpy(ref_data).float()
    with torch.no_grad():
        acts_ref = ref_sae.encode(ref_latents.to(eval_device)).cpu()

    if args.get("imglist_path") is not None:
        data_dir = Path(args["data_dir"])
        imglist_pth = Path(args["imglist_path"])
    else:
        if args["ind_name"] == "imagenet_mixed10_balanced":
            ind_tag = "mixed10_balanced"
            imglist_folder = "imagenet200"
        elif args["ind_name"] == "imagenet20":
            ind_tag = args["ind_name"]
        elif args["ind_name"] == "imagenet":
            ind_tag = args["ind_name"]
            imglist_folder = "imagenet"
        data_dir = Path("/gpfs/space/projects/mlgroup/data/images_largescale/")
        imglist_pth = Path(f'/gpfs/space/projects/mlgroup/data/benchmark_imglist/{imglist_folder}/{args["eval_split"]}_{ind_tag}.txt')
    try:
        _plot_two_heatmaps_with_acc_and_images(
                matrix_top=pre_losses,
                matrix_mid=None,
                matrix_bottom=ckas,
                accs=accs,                 # len == len(epochs)
                epochs=epochs,
                layers=layers,
                titles=[f"Ref: e{ref_epoch} l{ref_layer}.","Pre-Loss","CKA"],
                save_path= plots_dir / f"losses_ckas_ref_e{ref_epoch}_l{ref_layer}.png",
            )
    except:
        print("No Losses nor CKAs recorded")

    first_iter_flag = True

    if args.get("concept_types", False):
        from src.concept_type_classifier import run_concept_classification
        activation_threshold = 0.0

        plots_dir_2 = Path(args["plots_dir"]).parent / "concept_types"
        _ensure_dir(plots_dir_2)

    # Existing code continues here.
        for activation_threshold in [0.0,0.1,0.2]:
            concept_types, class_distributions = run_concept_classification(
                activations=acts_ref.T,
                labels_txt_path = imglist_pth,
                activation_threshold=activation_threshold,
                drop_ratio=0.5,
                max_superclass_size=10,
                ancestor_depth=1,
                imagenet_mapping_cache_path="imagenet_class_index.json",
            )
            print(concept_types)
            print("concept_types shape:", concept_types.shape)
            print("class_distributions shape:", class_distributions.shape)

            np.savez(plots_dir_2 / f"a_concept_types_e{ref_epoch}_l{ref_layer}_{activation_threshold}.npz", concept_types = concept_types, concept_dists =class_distributions)

    if args.get("combined_images", False):
        for i in range(total_nr_concepts):
            acts_ref_i = acts_ref.T[i]
            concept_i = concepts_stats[i]
            if i > 60:
                break
            if (np.sum(cor_res[i]) == 0) or (sum(acts_ref_i > 0) < 15) or (concept_i[0] < 0.001) or (max(acts_ref_i) < 0.2) or (concept_i[-1] < 1):
                continue



            titles = [f"Ref: e{ref_epoch} l{ref_layer}. Sparsity {np.round(concept_i[0],3)}; Mean  act. {np.round(concept_i[1],3)}; Lab. entropy {np.round(concept_i[2],3)}",
                    f"Cosine similarity {method}",
                    f"Spearman Correlation similarity {method}",
                    f"TopK similarity {method} ",
                        f"Noise of Correlation {method}"]
            act_buf = []
            nr_zeros = sum(acts_ref_i <= 0)
            non_zeros = acts_ref_i.argsort()[nr_zeros:]
            low = non_zeros[:5]
            act_buf.extend(acts_ref_i[low])
            high = non_zeros[-5:]
            mid_point = len(non_zeros)//2
            mid = non_zeros[mid_point-2:mid_point+3]
            act_buf.extend(acts_ref_i[mid])
            act_buf.extend(acts_ref_i[high])
            img_indices = []
            img_indices.extend(low)
            img_indices.extend(mid)
            img_indices.extend(high)

            _plot_two_heatmaps_with_acc_and_images(
                matrix_top=cor_res[i],
                matrix_mid=None,
                matrix_bottom=top_res[i],
                accs=accs,                 # len == len(epochs)
                epochs=epochs,
                layers=layers,
                titles=[titles[0],titles[2],titles[-2]],
                save_path= plots_dir / f"{method}_run_{run_idx}_ref_e{ref_epoch}_l{ref_layer}_concept{i}.png",
                img_indices=img_indices,
                data_dir=data_dir,
                imglist_pth=imglist_pth,
                cmap="viridis",
                acts_values = act_buf
            )
            if args["test_patches"]:
                ref_patch_latents =torch.from_numpy(ref_data[args["eval_split"]]["patches"]).float()
            else:
                ref_patch_latents = None

            if want_images:
                plot_top_activating_images_per_class(
                    num_classes=args["num_classes"],
                    activations=acts_ref_i, #None,  # shape [N]
                    sae = ref_sae,
                    ranking_source=args.get("topk_ranking","cls"),
                    feature_idx = i,
                    patch_latents = ref_patch_latents,
                    patch_grid = (14,14),
                    sae_device = eval_device,
                    save_path=plots_dir / f"class_dist_run_{run_idx}_ref_e{ref_epoch}_l{ref_layer}_concept{i}.png",
                    data_dir=data_dir,
                    imglist_pth=imglist_pth,
                    top_k=10,
                    preview_max_side=96,
                    title=f"Concept {i}: top activating images per class, sorted",
                )
                


   

def _evaluate_run(
    run_idx: int,
    args: Dict[str, Any],
    epochs: List[int],
    layers: List[int],
    ref_epoch: int,
    ref_layer: int,
    cache: Optional[Dict[Tuple[int, int], Dict[str, Any]]],
) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    eval_device = _device_or_default(args.get("eval_device"))
    #align_scores = np.full((len(layers), len(epochs)), np.nan, dtype=float)
    topk_scores = np.full((len(layers), len(epochs)), np.nan, dtype=float)
    #align_summaries: List[List[Dict[str, Any]]] = [[{} for _ in epochs] for _ in layers]
    criterion = torch.nn.MSELoss(reduction="mean")
    ref_ckpt = _checkpoint_path(args, ref_epoch, ref_layer, run_idx)
    if not ref_ckpt.exists():
        raise FileNotFoundError(f"Missing reference SAE checkpoint: {ref_ckpt}")
    ref_sae = _load_sae_for_eval(ref_ckpt, eval_device)

    ref_data = _load_all_data_cached(ref_epoch, ref_layer, args, cache, eval_mode=True)
    ref_latents = torch.from_numpy(ref_data).float().to(eval_device)
    with torch.no_grad():
        ref_sae_latents = ref_sae.encode(ref_latents).cpu()
    results_dir = Path(args["results_dir"]) / f"run_{run_idx}"
    if args["fixed_sae"]:
        results_dir = results_dir / f"fixed_sae{args['tag']}"
    if args.get("finetune_sae",False):
        results_dir = results_dir / "finetuned_sae"
        file_name = results_dir / f"similarity_matrices_run_{run_idx}_e{ref_epoch}_l{ref_layer}.npz"
        if not file_name.exists():
            ref_sae_latents,_, ref_sae = _finetune_sae(args,ref_ckpt,ref_epoch,ref_layer,cache)       
    _ensure_dir(results_dir)
    for li, layer in enumerate(layers):
        print("Layer:", layer)
        for ei, epoch in tqdm(enumerate(epochs),desc=f"layer {layer}"):
            file_name = results_dir / f"similarity_matrices_run_{run_idx}_e{epoch}_l{layer}.npz"
            if file_name.exists():
                print(f"skipped e{epoch} l{layer}")
                continue

            data = _load_all_data_cached(epoch, layer, args, cache,eval_mode = True)
            pre_loss, post_loss = None, 0
            ckpt_latents = torch.from_numpy(data).float().to(eval_device)
            if args["fixed_sae"]:
                with torch.no_grad():
                    acts_a, recon = ref_sae.encode(ckpt_latents, only_z=False)
                recon_loss = criterion(recon, ckpt_latents)
                pre_loss = recon_loss.item()

                if args.get("finetune_sae",False):
                    #####finetune 1 epoch
                    if (ei,li) != (ref_epoch,ref_layer):
                        acts_a, post_loss, _ = _finetune_sae(args,ref_ckpt,ei,li,cache)
                    else:
                        post_loss = pre_loss
                        acts_a = ref_sae_latents

            else:
                ckpt = _checkpoint_path(args, epoch, layer, run_idx)
                if not ckpt.exists():
                    raise FileNotFoundError(f"Missing SAE checkpoint: {ckpt}")
                sae = _load_sae_for_eval(ckpt, eval_device)
                with torch.no_grad():
                    acts_a = sae.encode(ckpt_latents)

            cka = linear_cka(ref_latents, ckpt_latents)
            print("pre_loss:", pre_loss, "post_loss", post_loss, "cka", cka)

           
            sim_res_dict = get_saes_similarities(
                acts_a.to(eval_device),
                ref_sae_latents.to(eval_device),
                args["fixed_sae"] == args.get("finetune_sae",False),
                return_cos =True,
                method=args["align_method"],
                min_sim=args["align_min_sim"],
                compute_stability=args["align_compute_stability"],
                compute_chance=args["align_compute_chance"],
                device=eval_device,
                return_similarity = True,
                k=args["topk_k"],
                metric=args["topk_metric"],
                threshold=args["topk_threshold"],
                act_threshold=args["topk_act_threshold"]
            )
            #align_summary = sim_res_dict["hungarian_res"]["summary"]
            #align_scores[li, ei] = float(align_summary.get(args["align_summary_key"], np.nan))
            #align_summaries[li][ei] = _json_safe(align_summary)
            topk_scores[li, ei] = sim_res_dict["topk_sim"].max(1).mean()
            np.savez(
                file_name,
                sim_cos=sim_res_dict["cos_sim"],
                sim_topk=sim_res_dict["topk_sim"],
                sim_cor=sim_res_dict["cor_sim"],
                random_cos = sim_res_dict["cos_random"],
                random_cor = sim_res_dict["cor_random"],
                losses = [pre_loss, post_loss],
                cka = cka,
                #hungary_match_scores = sim_res_dict["hungarian_res"]
            )
            del sim_res_dict, data, ckpt_latents
            torch.cuda.empty_cache()

    return topk_scores 


def _save_run_results(
    run_idx: int,
    args: Dict[str, Any],
    epochs: List[int],
    layers: List[int],
    ref_epoch: int,
    ref_layer: int,
    topk_scores: np.ndarray,
) -> None:
    results_dir = Path(args["results_dir"]) / f"run_{run_idx}"
    _ensure_dir(results_dir)

    np.savez(
        results_dir / "similarity_matrices.npz",
        topk=topk_scores,
        epochs=np.array(epochs),
        layers=np.array(layers),
        ref_epoch=np.array([ref_epoch]),
        ref_layer=np.array([ref_layer]),
    )
    
    plots_dir = Path(args["plots_dir"]) / f"run_{run_idx}"
    _plot_heatmap(
        topk_scores,
        epochs,
        layers,
        f"TopK Sets ({args['topk_summary_key']}) - run {run_idx} ref: epoch {ref_epoch} layer{ref_layer}",
        plots_dir / f"topk_{args['topk_summary_key']}_run_{run_idx}_ref_e{ref_epoch}_l{ref_layer}.png",
    )

def _save_concept_plots(
    run_idx: int,
    args: Dict[str, Any],
    epochs: List[int],
    layers: List[int],
    ref_epoch: int,
    ref_layer: int,
    method : str = "max",
    want_images : bool = True,
    eval_device = None
) -> None:
    results_dir = Path(args["results_dir"]) / f"run_{run_idx}"
    if args["fixed_sae"]:
        results_dir = results_dir / f"fixed_sae{args['tag']}"
    if args.get("finetune_sae",False):
        results_dir = results_dir / "finetuned_sae"
    #sim_res_dict = np.load(results_dir / f"similarity_matrices_run_{run_idx}_e50_l0.npz")["cor_sim"]
    total_nr_concepts = args["vit_size"]//4*args["expansion_coef"]
    cos_res = np.full((total_nr_concepts ,len(layers), len(epochs)), np.nan, dtype=float)
    cor_res = np.full((total_nr_concepts ,len(layers), len(epochs)), np.nan, dtype=float)
    cos_res_noise = np.full((total_nr_concepts ,len(layers), len(epochs)), np.nan, dtype=float)
    cor_res_noise = np.full((total_nr_concepts ,len(layers), len(epochs)), np.nan, dtype=float)
    top_res = np.full((total_nr_concepts ,len(layers), len(epochs)), np.nan, dtype=float)

    loss_diff = np.full((len(layers), len(epochs)), np.nan, dtype=float)
    pre_losses = np.full((len(layers), len(epochs)), np.nan, dtype=float)
    ckas = np.full((len(layers), len(epochs)), np.nan, dtype=float)


    for li, layer in enumerate(layers):
        for ei, epoch in enumerate(epochs):
            #print(ei,li)
            sim_res_dict = np.load(results_dir / f"similarity_matrices_run_{run_idx}_e{epoch}_l{layer}.npz")
            #hungary_match_scores = sim_res_dict["hungarian_res"]
             
            if args["fixed_sae"]:
                cos_res[:,li,ei] = sim_res_dict["sim_cos"].diagonal(0) 
                cor_res[:,li,ei] = sim_res_dict["sim_cor"].diagonal(0)
                top_res[:,li,ei] = sim_res_dict["sim_topk"].diagonal(0)

            elif method == "max":
                cos_res[:,li,ei] = sim_res_dict["sim_cos"].max(0) - sim_res_dict["random_cos"]
                cor_res[:,li,ei] = sim_res_dict["sim_cor"].max(0) - sim_res_dict["random_cor"]
                top_res[:,li,ei] = sim_res_dict["sim_topk"].max(0)

            cos_res_noise[:,li,ei] =  sim_res_dict["random_cos"]
            cor_res_noise[:,li,ei] =  sim_res_dict["random_cor"]
            try:
                loss_diff[li,ei] = sim_res_dict["losses"][0] - sim_res_dict["losses"][1]
                pre_losses[li,ei] = sim_res_dict["losses"][0]
            except:
                print("no losses")
            try:
                ckas[li,ei] = sim_res_dict["cka"]
            except:
                print("no ckas")
    try:
        loss_diff = (loss_diff - np.min(loss_diff))/(np.max(loss_diff) - np.min(loss_diff))
    except:
        print("noting")

      
            
    plots_dir = Path(args["plots_dir"]) / f"run_{run_idx}"
    if args["fixed_sae"]:
        plots_dir = plots_dir / f"fixed_sae{args['tag']}"
    if args.get("finetune_sae",False):
        plots_dir = plots_dir / "finetuned_sae"
    _ensure_dir(plots_dir)

    accs = []
    if not args["pretrained_vit_IN1K"]:
        if args["num_heads"] == 3:
            log_path = Path(f'../openood/OpenOOD/results/{args["ind_name"]}_vit-b-16_base_e{args["total_vit_epochs"]}_lr0.1_default/s{args["seed"].split("s")[-1]}/log.txt')
            with log_path.open("r") as f:
                for line in f:
                    if "Val Acc" in line:
                        accs.append(float(line.split("Acc")[1].strip()))
        else:
            if args["seed"] == "_s8":
                log_path = Path(f'../openood/OpenOOD/results/vit_tiny_ffcv/log.out')
            elif args["seed"] == "_s7":
                log_path = Path(f'../openood/OpenOOD/results/vit_tiny_a1_mixed/log.out')
            with log_path.open("r") as f:
                for line in f:
                    if "val_acc" in line:
                        accs.append(float(line.split("val_acc1=")[1].split(" ")[0].strip()))
        accs = np.array(accs)[epochs]
    else:
        accs = np.array([81])


    ref_ckpt = _checkpoint_path(args, ref_epoch, ref_layer, run_idx)
    if not ref_ckpt.exists():
        raise FileNotFoundError(f"Missing reference SAE checkpoint: {ref_ckpt}")
    ref_sae = _load_sae_for_eval(ref_ckpt, eval_device)
    cache = {} if args["cache_activations"] else None
    ref_data = _load_all_data_cached(ref_epoch, ref_layer, args, cache)
    ref_labels = torch.from_numpy(ref_data[args["eval_split"]]["labels"])
    ref_latents = torch.from_numpy(ref_data[args["eval_split"]]["activations"]).float()
    ref_latents_tr = torch.from_numpy(ref_data[args["train_split"]]["activations"]).float()
    with torch.no_grad():
        acts_ref = ref_sae.encode(ref_latents.to(eval_device)).cpu()
        acts_ref_tr = ref_sae.encode(ref_latents_tr.to(eval_device)).cpu()

    concepts_stats = concept_summary_stats(acts_ref, ref_labels, top_k=None)
    stats_np = []
    for key in concepts_stats:
        c = concepts_stats[key]
        stats_np.append([c["sparsity"],c["mean_activation"],c["label_entropy"],c["label_std"]])

    np.savez(plots_dir / "evolution_patterns.npz",
    cos_res = cos_res, cor_res = cor_res, top_res =top_res,concepts = np.array(stats_np))

    plot_metrics_figure(concepts_stats, save_path = plots_dir / f"concept_stats_run_{run_idx}_ref_e{ref_epoch}_l{ref_layer}.png")
    
    criteria = {
    "sparsity":        (10**(-2.0), None),   # upper-bounded
    "label_entropy":   (None,None),    # upper-bounded
    "mean_activation": (10**(-1.2), None),   # lower-bounded
        }

    if args["ind_name"] == "imagenet_mixed10_balanced":
        ind_tag = "mixed10_balanced"
        imglist_folder = "imagenet200"
    elif args["ind_name"] == "imagenet20":
        ind_tag = args["ind_name"]
    elif args["ind_name"] == "imagenet":
        ind_tag = args["ind_name"]
        imglist_folder = "imagenet"
    try:
        _plot_two_heatmaps_with_acc_and_images(
                matrix_top=pre_losses,
                matrix_mid=loss_diff,
                matrix_bottom=ckas,
                accs=accs,                 # len == len(epochs)
                epochs=epochs,
                layers=layers,
                titles=[f"Ref: e{ref_epoch} l{ref_layer}.","Pre-Loss","Loss dif","CKA"],
                save_path= plots_dir / f"losses_ckas_ref_e{ref_epoch}_l{ref_layer}.png",
            )
    except:
        print("No Losses nor CKAs recorded")

    first_iter_flag = True
    for i in range(total_nr_concepts):
        acts_ref_i = acts_ref.T[i]

        if (np.sum(cor_res[i]) == 0) or (sum(acts_ref_i > 0) < 15):
            continue

        concept_i = concepts_stats[i]

        titles = [f"Ref: e{ref_epoch} l{ref_layer}. Sparsity {np.round(concept_i['sparsity'],3)}; Mean  act. {np.round(concept_i['mean_activation'],3)}; Lab. entropy {np.round(concept_i['label_entropy'],3)}; Lab. std {np.round(concept_i['label_std'],3)}",
                f"Cosine similarity {method}",
                f"Correlation similarity {method}",
                f"TopK similarity {method} ",
                 f"Noise of Correlation {method}"]
    
        nr_zeros = sum(acts_ref_i <= 0)
        non_zeros = acts_ref_i.argsort()[nr_zeros:]
        low = non_zeros[:5]
        high = non_zeros[-5:]
        mid_point = len(non_zeros)//2
        mid = non_zeros[mid_point-2:mid_point+3]
        img_indices = []
        img_indices.extend(low)
        img_indices.extend(mid)
        img_indices.extend(high)

        _plot_two_heatmaps_with_acc_and_images(
            matrix_top=cor_res[i],
            matrix_mid=cor_res_noise[i],
            matrix_bottom=top_res[i],
            accs=accs,                 # len == len(epochs)
            epochs=epochs,
            layers=layers,
            titles=[titles[0],titles[2],titles[-1],titles[-2]],
            save_path= plots_dir / f"{method}_run_{run_idx}_ref_e{ref_epoch}_l{ref_layer}_concept{i}.png",
            img_indices=img_indices,
            data_dir=Path("/gpfs/space/projects/mlgroup/data/images_largescale/"),
            imglist_pth=Path(f'/gpfs/space/projects/mlgroup/data/benchmark_imglist/{imglist_folder}/{args["eval_split"]}_{ind_tag}.txt'),
            cmap="viridis",
        )
        if args["test_patches"]:
            ref_patch_latents =torch.from_numpy(ref_data[args["eval_split"]]["patches"]).float()
        else:
            ref_patch_latents = None
        #with torch.no_grad():
        #    if args["test_patches"]:
        #        acts_ref_patches = ref_sae.encode(ref_patch_latents.reshape((-1,args["vit_size"]//4)).to(eval_device)).cpu().reshape((ref_latents.shape[0],-1,ref_latents.shape[1]))
        #    else:
        #        acts_ref_patches = None


        if want_images:
            plot_top_activating_images_per_class(
                num_classes=args["num_classes"],
                activations=acts_ref_i, #None,  # shape [N]
                sae = ref_sae,
                ranking_source=args.get("topk_ranking","cls"),
                feature_idx = i,
                patch_latents = ref_patch_latents,
                patch_grid = (14,14),
                sae_device = eval_device,
                save_path=plots_dir / f"class_dist_run_{run_idx}_ref_e{ref_epoch}_l{ref_layer}_concept{i}.png",
                data_dir=Path("/gpfs/space/projects/mlgroup/data/images_largescale/"),
                imglist_pth=Path(f'/gpfs/space/projects/mlgroup/data/benchmark_imglist/{imglist_folder}/{args["eval_split"]}_{ind_tag}.txt'),
                top_k=10,
                preview_max_side=96,
                title=f"Concept {i}: top activating images per class",
            )
        
        if args["probing"]:
            savepath = plots_dir / "probing"
            _ensure_dir(savepath)
            if first_iter_flag:

                print("here1")
                first_iter_flag = False
                te_lab_pth=Path(f'../openood/OpenOOD/data/benchmark_imglist/imagenet200/{args["eval_split"]}_{ind_tag}.json')
                tr_lab_pth=Path(f'../openood/OpenOOD/data/benchmark_imglist/imagenet200/{args["train_split"]}_{ind_tag}.json')
                with open(te_lab_pth, 'r') as f:
                    te_lab_dict = json.load(f)
                with open(tr_lab_pth, 'r') as f:
                    tr_lab_dict = json.load(f)
                
                test_subclass_names = []
                for k,v in te_lab_dict.items():
                    test_subclass_names.append(v[2])

                train_subclass_names = []
                for k,v in tr_lab_dict.items():
                    train_subclass_names.append(v[2])
                subclass_names = np.unique(np.array(train_subclass_names))
                print(subclass_names)

                shared_df = precompute_shared_subclass_metrics(
                            subclass_names=subclass_names,
                            subclass_labels_train=train_subclass_names,
                            subclass_labels_test=test_subclass_names,
                            full_activation_train=ref_latents_tr,
                            full_activation_test=ref_latents,
                            full_sae_train=acts_ref_tr,
                            full_sae_test=acts_ref,
                            save_path= savepath / f"Probing_reference.csv",
                            probe_C=1.0,
                            )
            plot_subclass_probe_heatmap_cached(
                    shared_df=shared_df,
                    subclass_names=subclass_names,
                    subclass_labels_train=train_subclass_names,
                    subclass_labels_test=test_subclass_names,
                    sae_concept_train=acts_ref_tr[:, i],
                    sae_concept_test=acts_ref[:, i],
                    save_path= savepath/ f"Probing_run_{run_idx}_ref_e{ref_epoch}_l{ref_layer}_concept{i}.png",
                    activation_threshold=0.0,
                    title=f"Concept {i}",
                )
            




def _save_average_results(
    args: Dict[str, Any],
    epochs: List[int],
    layers: List[int],
    ref_epoch: int,
    ref_layer: int,
    all_topk: np.ndarray,
) -> None:
    results_dir = Path(args["results_dir"])
    _ensure_dir(results_dir)
    np.savez(
        results_dir / "all_runs_matrices.npz",
        topk=all_topk,
        epochs=np.array(epochs),
        layers=np.array(layers),
        ref_epoch=np.array([ref_epoch]),
        ref_layer=np.array([ref_layer]),
    )
    #mean_align = np.nanmean(all_align, axis=0)
    mean_topk = np.nanmean(all_topk, axis=0)
    np.savez(
        results_dir / "mean_matrices.npz",
        topk=mean_topk,
        epochs=np.array(epochs),
        layers=np.array(layers),
        ref_epoch=np.array([ref_epoch]),
        ref_layer=np.array([ref_layer]),
    )

    plots_dir = Path(args["plots_dir"]) / "average"
    _plot_heatmap(
        mean_topk,
        epochs,
        layers,
        f"TopK Sets ({args['topk_summary_key']}) - mean over runs_ref: epoch {ref_epoch} layer{ref_layer}",
        plots_dir / f"topk_{args['topk_summary_key']}_mean_ref_e{ref_epoch}_l{ref_layer}.png",
    )

def make_epoch_grid(ref_epoch, n_epochs=300, half_window=50, low_stride=10):
    full_window = 2 * half_window

    # Clamp the high-resolution window so it always has length 100
    start = ref_epoch - half_window
    start = max(0, start)
    start = min(start, n_epochs - full_window)

    end = start + full_window  # exclusive

    high_res = np.arange(start, end, 1)

    low_left = np.arange(0, start, low_stride)
    low_right = np.arange(end, n_epochs, low_stride)

    epochs = np.unique(np.concatenate([low_left, high_res, low_right]))
    epochs.sort()

    return epochs

def main(args) -> None:
    import yaml
    import io

    if args.config == None:
        with open("sae_exp_config_base.yml", 'r') as stream:
            args_dict = yaml.safe_load(stream)
        print("using base")
    else:
        with open(args.config, 'r') as stream:
            args_dict = yaml.safe_load(stream)       
        args_dict["plots_dir"] = "figs/sae_similarity"
        args_dict["results_dir"] = "eval_results"
        args_dict["checkpoint_folder"] = "sae_checkpoints"
        print("using path")
    if args.rl != None:
        args_dict["ref_epoch"] = int(args.re)
        args_dict["ref_layer"] = int(args.rl)

    ref_epoch = args_dict["ref_epoch"] #0
    ref_layer = args_dict["ref_layer"] #11
    print('epoch', ref_epoch,'layer', ref_layer)

    args_dict["checkpoint_folder"] = f'{args_dict["checkpoint_folder"]}/{args_dict["backbone"]}_{args_dict["ind_name"]}{args_dict["seed"]}/refepoch_{ref_epoch}_reflayer_{ref_layer}'
    checkpoint_folder = Path(args_dict["checkpoint_folder"])
    _ensure_dir(checkpoint_folder)
    
    args_dict["plots_dir"] = f'{args_dict["plots_dir"]}/{args_dict["backbone"]}_{args_dict["ind_name"]}{args_dict["seed"]}/refepoch_{ref_epoch}_reflayer_{ref_layer}'
    plots_dir = Path(args_dict["plots_dir"])
    if args_dict["fixed_sae"]:
        plots_dir = plots_dir / f"fixed_sae{args_dict['tag']}"
    _ensure_dir(plots_dir)

    args_dict["results_dir"] = f'{args_dict["checkpoint_folder"]}/{args_dict["results_dir"]}'
    results_dir = Path(args_dict["results_dir"])
    _ensure_dir(results_dir)
    ## for saving the config in better place
    if args_dict["fixed_sae"]:
        buf_results_dir = results_dir/ "run_0" / f"fixed_sae{args_dict['tag']}"
    else:
        buf_results_dir = results_dir/"run_0"
    if args_dict.get("finetune_sae",False):
        buf_results_dir = results_dir / "run_0" / f"fixed_sae{args_dict['tag']}"/ "finetuned_sae"
    _ensure_dir(buf_results_dir)

    with io.open(buf_results_dir / 'config.yaml', 'w', encoding='utf8') as outfile:
        yaml.dump(args_dict, outfile, default_flow_style=False, allow_unicode=True)

    epochs = list(range(args_dict["vit_epochs"][0],args_dict["vit_epochs"][1]))
    layers = list(range(args_dict["layer_nrs"][0],args_dict["layer_nrs"][1]))

    cache = {} if args_dict["cache_activations"] else None
    eval_device = _device_or_default(args_dict.get("eval_device"))
    if args_dict["train_saes"]:
        _train_all_saes(args_dict, epochs, layers,ref_epoch, ref_layer, cache)

    all_topk = np.full((args_dict["sae_runs"], len(layers), len(epochs)), np.nan, dtype=float)



    print(eval_device)
    for run_idx in range(args_dict["sae_runs"]):
        
        if args_dict["ind_name"] == "shapes3d":
            ref_ckpt = _checkpoint_path(args_dict, ref_epoch, ref_layer, run_idx)
            if not ref_ckpt.exists():
                raise FileNotFoundError(f"Missing reference SAE checkpoint: {ref_ckpt}")

            ref_sae = _load_sae_for_eval(ref_ckpt, eval_device)
            ref_data = _load_all_data_cached(ref_epoch, ref_layer, args_dict, cache)
        
        if not args_dict["only_plots"]:
#            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, ) as prof:
#                with record_function("model_inference")
            
            if args_dict.get("combined",False):
                ###implement the epochs and layers windwonf
                half_window = 50      # because total window is 100
                n_epochs = args_dict["total_vit_epochs"]        # epochs are assumed to be 0, ..., 299
                low_stride = 10

                epochs_windowed =  make_epoch_grid(ref_epoch, n_epochs, half_window, low_stride)
                print(epochs_windowed)
                results_dir = Path(args_dict["results_dir"]).parent.parent / "evo_patterns"
                evo_pat_path = results_dir / f"evolution_patterns_e{ref_epoch}_l{ref_layer}.npz"
                if not evo_pat_path.exists():

                    _evaluate_run_combined(
                            run_idx,
                            args_dict,
                            epochs_windowed,
                            layers,
                            ref_epoch,
                            ref_layer,
                            cache,  
                        )
                else:
                    print("CKPT existed")
                _save_concept_evo_pattern(run_idx ,args_dict, epochs_windowed, layers, ref_epoch, ref_layer, "max", args_dict["want_images"], eval_device)

                break

            else:
                topk_scores = _evaluate_run(
                        run_idx,
                        args_dict,
                        epochs,
                        layers,
                        ref_epoch,
                        ref_layer,
                        cache,  
                    )

            #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=50))
            #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))


            #all_align[run_idx] = align_scores
            all_topk[run_idx] = topk_scores
            _save_run_results(
                run_idx,
                args_dict,
                epochs,
                layers,
                ref_epoch,
                ref_layer,
                topk_scores,
            )

            _save_average_results(args_dict, epochs, layers, ref_epoch, ref_layer,all_topk)
        if eval_device == "cpu":
            _save_concept_plots(run_idx ,args_dict, epochs, layers, ref_epoch, ref_layer, "max", args_dict["want_images"], eval_device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Loading args")
    parser.add_argument('--config',help = "config path", default = None)
    parser.add_argument('--re', help = "ref_epoch",default = None)
    parser.add_argument('--rl', help = "ref_layer",default = None)
    args = parser.parse_args()
    main(args)
