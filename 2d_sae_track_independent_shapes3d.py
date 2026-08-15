from __future__ import annotations
import os
os.environ["SCIPY_ARRAY_API"] = "1"
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import argparse
from tqdm import tqdm

from src.base_sae import BatchTopKSparseAutoencoder, SparseAutoencoder, TopKSparseAutoencoder
from src.sae_utils import  load_checkpoint, save_checkpoint, train_sae, get_saes_similarities, align_concepts_hungarian, concept_summary_stats, plot_metrics_figure, query_concepts, linear_cka, train_sae_fast
from src.utils import load_all_data, _tokens_to_patch_map, _infer_patch_grid, _reduce_token_scores, _normalize_patch_map_for_display
from src.plotting import make_heatmap
from src.probing import plot_subclass_probe_heatmap_cached, precompute_shared_subclass_metrics
from torch.profiler import profile, record_function, ProfilerActivity
from src.concept_type_classifier import run_concept_classification


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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _device_or_default(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    key = (epoch, layer)
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
            eval_split = args["eval_split"]
        )
    except:
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


def _plot_heatmap(
    matrix: np.ndarray,
    epochs: List[int],
    layers: List[int],
    title: str,
    save_path: Path,
    *,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
    cmap: str = "viridis",
) -> None:
    fig_w = max(6.0, 0.7 * len(epochs))
    fig_h = max(4.0, 0.5 * len(layers))
    plt.figure(figsize=(fig_w, fig_h))
    vmax = _second_highest_value(matrix)
    sns.heatmap(
        matrix,
        xticklabels=[str(e) for e in epochs],
        yticklabels=[str(l) for l in layers],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar=True,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    _ensure_dir(save_path.parent)
    plt.savefig(save_path, dpi=300)
    plt.close()

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
    args: Dict[str, Any],
    epochs: List[int],
    layers: List[int],
    ref_epoch,
    ref_layer,
    cache ,
) -> None:
    train_device = _device_or_default(args.get("train_device"))

    for run_idx in range(args["sae_runs"]):
        seed = _run_seed(args["run_seed_base"], run_idx, ref_epoch, ref_layer)
        if args["fixed_sae"]:
            ckpt = _checkpoint_path(args, ref_epoch, ref_layer, run_idx)
            print(ckpt)
            if ckpt.exists():
                pass
            else:
                seed_everything(seed)
                data = _load_all_data_cached(ref_epoch, ref_layer, args, cache)
                if args["trained_on"] in ["cls", "mix", None]:
                    latents = torch.from_numpy(data[args["train_split"]]["activations"]).float()
                if args["trained_on"] in ["mix","patches"]:
                    patch_latents = torch.from_numpy(data[args["train_split"]]["patches"]).float().reshape((-1,args["vit_size"]//4))
                    if args["trained_on"] == "patches":
                        latents = patch_latents
                    else:
                        latents = torch.cat([latents, patch_latents],dim = 0)
                seed_everything(seed)
                sae = _build_sae(latents.shape[1], args)
                
                train_sae(
                    sae,
                    latents,
                    epochs=args["sae_epochs"],
                    batch_size=args["batch_size"],
                    lr=args["lr"],
                    l1_lambda=args["l1_lambda"],
                    z_l2_lambda=args["l2_lambda"],
                    verbose_every=args["verbose_every"],
                    device=train_device,
                )
                meta = {
                    "sae_type": getattr(sae, "sae_type", args["sae_type"]),
                    "k": getattr(sae, "k", None),
                    "by_abs": getattr(sae, "by_abs", False),
                    "expansion_coef": args["expansion_coef"],
                    "epoch": ref_epoch,
                    "layer": ref_layer,
                    "run": run_idx,
                    "seed": args["seed"],
                    "step": args["step"],
                    "backbone": args["backbone"],
                    "trained_on" : args["trained_on"], #"cls, patch, mixed"
                }
                save_checkpoint(sae, ckpt, meta=meta)
            return 
    for run_idx in range(args["sae_runs"]):
        print(f"[train] run {run_idx + 1}/{args['sae_runs']}")
        seed = _run_seed(args["run_seed_base"], run_idx, ref_epoch, ref_layer)
        for epoch in epochs:            
            for layer in layers:
                ckpt = _checkpoint_path(args, epoch, layer, run_idx)
                if args["skip_existing"] and ckpt.exists():
                    continue
                #seed_everything(_run_seed(args["run_seed_base"], run_idx, epoch, layer))
                seed_everything(seed)
                data = _load_all_data_cached(epoch, layer, args, cache)
                if args["trained_on"] in ["cls", "mix", None]:
                    latents = torch.from_numpy(data[args["train_split"]]["activations"]).float()
                if args["trained_on"] in ["mix","patches"]:
                    patch_latents = torch.from_numpy(data[args["train_split"]]["patches"]).float().reshape((-1,args["vit_size"]//4))
                    if args["trained_on"] == "patches":
                        latents = patch_latents
                    else:
                        latents = torch.cat([latents, patch_latents],dim = 0)
                seed_everything(seed)
                sae = _build_sae(latents.shape[1], args)
                if (epoch,layer) == (ref_epoch,ref_layer):
                    train_sae(
                        sae,
                        latents,
                        epochs=args["sae_epochs"],
                        batch_size=args["batch_size"],
                        lr=args["lr"],
                        l1_lambda=args["l1_lambda"],
                        z_l2_lambda=args["l2_lambda"],
                        verbose_every=args["verbose_every"],
                        device=train_device,
                    )
                else:
                    train_sae_fast(
                        sae,
                        latents,
                        epochs=args["sae_epochs"],
                        batch_size=args["batch_size"],
                        lr=args["lr"],
                        l1_lambda=args["l1_lambda"],
                        z_l2_lambda=args["l2_lambda"],
                        verbose_every=args["verbose_every"],
                        device=train_device,
                    )
                    
                meta = {
                    "sae_type": getattr(sae, "sae_type", args["sae_type"]),
                    "k": getattr(sae, "k", None),
                    "by_abs": getattr(sae, "by_abs", False),
                    "expansion_coef": args["expansion_coef"],
                    "epoch": epoch,
                    "layer": layer,
                    "run": run_idx,
                    "seed": args["seed"],
                    "step": args["step"],
                    "backbone": args["backbone"],
                    "trained_on" : args["trained_on"], #"cls, patch, mixed"
                }
                save_checkpoint(sae, ckpt, meta=meta)
                
 
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

    total_nr_concepts = args["vit_size"]//4*args["expansion_coef"]
    
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
                acts_a, recon = ref_sae.encode(ckpt_latents, only_z=False)
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


            #print("pre_loss:", pre_loss, "post_loss", post_loss, "cka", cka)
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
    ref_data = _load_all_data_cached(ref_epoch, ref_layer, args, cache, eval_mode=True)
    ref_latents = torch.from_numpy(ref_data).float()
    with torch.no_grad():
        acts_ref = ref_sae.encode(ref_latents.to(eval_device)).cpu()

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
        activation_threshold = 0.0

        plots_dir_2 = Path(args["plots_dir"]).parent / "concept_types"
        _ensure_dir(plots_dir_2)

        for activation_threshold in [0.0,0.1,0.2]:
            concept_types, class_distributions = run_concept_classification(
                activations=acts_ref.T,
                labels_txt_path = Path(f'/gpfs/space/projects/mlgroup/data/benchmark_imglist/{imglist_folder}/{args["eval_split"]}_{ind_tag}.txt'),
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
                data_dir=Path("/gpfs/space/projects/mlgroup/data/images_largescale/"),
                imglist_pth=Path(f'/gpfs/space/projects/mlgroup/data/benchmark_imglist/{imglist_folder}/{args["eval_split"]}_{ind_tag}.txt'),
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
                    data_dir=Path("/gpfs/space/projects/mlgroup/data/images_largescale/"),
                    imglist_pth=Path(f'/gpfs/space/projects/mlgroup/data/benchmark_imglist/{imglist_folder}/{args["eval_split"]}_{ind_tag}.txt'),
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
    
    #with open(results_dir / "align_summaries.json", "w", encoding="utf-8") as f:
    #    json.dump(_json_safe(align_summaries), f, indent=2)


    plots_dir = Path(args["plots_dir"]) / f"run_{run_idx}"
    #_plot_heatmap(
    #    align_scores,
    #    epochs,
    #    layers,
    #    f"Align Hungarian ({args['align_summary_key']}) - run {run_idx} ref: epoch {ref_epoch} layer {ref_layer}",
    #    plots_dir / f"align_{args['align_summary_key']}_run_{run_idx}_ref_e{ref_epoch}_l{ref_layer}.png")

    _plot_heatmap(
        topk_scores,
        epochs,
        layers,
        f"TopK Sets ({args['topk_summary_key']}) - run {run_idx} ref: epoch {ref_epoch} layer{ref_layer}",
        plots_dir / f"topk_{args['topk_summary_key']}_run_{run_idx}_ref_e{ref_epoch}_l{ref_layer}.png",
    )

def _second_highest_value(matrix: np.ndarray) -> float:
    """Second-highest *distinct* finite value (fallback to max if not available)."""
    vals = np.asarray(matrix).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    uniq = np.unique(vals)
    uniq.sort()
    return float(uniq[-2] if uniq.size >= 2 else uniq[-1])


def _plot_three_heatmaps(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    matrix3: np.ndarray,
    accs: Sequence[float],
    epochs: List[int],
    layers: List[int],
    titles: Tuple[str, str, str],
    save_path: Path,
    *,
    vmin: Optional[float] = 0.0,
    cmap: str = "viridis",
    acc_ylabel: str = "Accuracy",
    acc_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    if len(accs) != len(epochs):
        raise ValueError(f"len(accs)={len(accs)} must match len(epochs)={len(epochs)}")
    
    fig_w = max(6.0, 0.7 * len(epochs))
    base_h = max(2.8, 0.5 * len(layers))
    acc_h = max(1.4, 0.35 * base_h)  # dedicated height for the accuracy strip
    fig_h = base_h * 3.0 + acc_h

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[acc_h, base_h, base_h, base_h])

    # --- accuracy axis (above the top heatmap) ---
    ax_acc = fig.add_subplot(gs[0, 0])

    # Heatmap axes (share x/y among themselves)
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax1, sharey=ax1)
    heat_axes = (ax1, ax2, ax3)

    # x positions that align with heatmap column centers (0.5, 1.5, ..., n-0.5)
    x = np.arange(len(epochs)) + 0.5
    ax_acc.plot(x, np.asarray(accs, dtype=float))
    ax_acc.set_xlim(0, len(epochs))
    ax_acc.set_ylabel(acc_ylabel)
    if acc_ylim is not None:
        ax_acc.set_ylim(*acc_ylim)
    ax_acc.grid(True, axis="y", alpha=0.3)
    ax_acc.tick_params(axis="x", labelbottom=False)  # keep epoch labels only on bottom heatmap

    # --- heatmaps ---
    for ax, mat, title in zip(heat_axes, (matrix1, matrix2, matrix3), titles):
        vmax = _second_highest_value(mat)
        sns.heatmap(
            mat,
            ax=ax,
            xticklabels=[str(e) for e in epochs],
            yticklabels=[str(l) for l in layers],
            vmin=min(mat.min(),0.0),
            vmax=vmax,
            cmap=cmap,
            cbar=True,
        )
        ax.set_title(title)
        ax.set_ylabel("Layer")

    heat_axes[-1].set_xlabel("Epoch")

    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _read_imglist(imglist_pth: Path) -> Tuple[List[str], List[int]]:
    """
    Reads ImglistDataset txt format: each line like '<rel_path> <label>'
    (path may occasionally have spaces; we treat last token as label).
    """
    rel_paths: List[str] = []
    labels: List[int] = []
    with open(imglist_pth, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            label = int(parts[-1])
            rel_path = " ".join(parts[:-1])
            rel_paths.append(rel_path)
            labels.append(label)
    return rel_paths, labels


def _resolve_image_path(data_dir: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (data_dir / p)

def _load_preview_image(
    img_path: Path,
    max_side: int = 96,
) -> Image.Image:
    """
    Load image and downsample it for fast plotting.
    max_side=96 or 128 is usually enough for subplot previews.
    """
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return im

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Optional, List


def _orient_matrix(mat: np.ndarray, n_layers: int, n_epochs: int, name: str) -> np.ndarray:
    mat = np.asarray(mat)

    if mat.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {mat.shape}")

    if mat.shape == (n_layers, n_epochs):
        return mat

    if mat.shape == (n_epochs, n_layers):
        print(f"[warning] {name} appears transposed; using {name}.T")
        return mat.T

    raise ValueError(
        f"{name} has shape {mat.shape}, but expected "
        f"({n_layers}, {n_epochs}) or ({n_epochs}, {n_layers})"
    )


def _plot_two_heatmaps_with_acc_and_images(
    matrix_top: np.ndarray,
    matrix_mid: np.ndarray,
    matrix_bottom: np.ndarray,
    accs: Sequence[float],
    epochs: List[int],
    layers: List[int],
    titles: Sequence[str],   # expected: [suptitle, top, mid, bottom]
    save_path: Path,
    *,
    cmap: str = "viridis",
    img_indices: Optional[Sequence[int]] = None,
    data_dir: Optional[Path] = None,
    imglist_pth: Optional[Path] = None,
    show_image_labels: bool = False,
    acc_ylabel: str = "Accuracy",
    acc_ylim: Optional[tuple[float, float]] = None,
    acts_values = None
) -> None:
    want_images = (
        img_indices is not None
        and len(img_indices) > 0
        and data_dir is not None
        and imglist_pth is not None
    )

    n_epochs = len(epochs)
    n_layers = len(layers)
    if type(matrix_mid) == type(None):
        matrices = [matrix_top, matrix_bottom]
    else:
        matrices = [matrix_top, matrix_mid, matrix_bottom]

    # --- saner figure geometry ---
    fig_w = min(22.0, max(10.0, 0.18 * n_epochs))
    heat_h = min(5.5, max(2.5, 0.28 * n_layers))
    acc_h = 1.6
    images_h = 2.2 if want_images else 0.0
    label_h = 0.5 if want_images else 0.0
    fig_h = acc_h + len(matrices) * heat_h + images_h + label_h

    nrows = (len(matrices) + 3) if want_images else (len(matrices) + 1)
    height_ratios = [acc_h] + [heat_h]*len(matrices) + ([images_h, label_h] if want_images else [])

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.suptitle(titles[0], fontsize=18)

    gs = fig.add_gridspec(nrows=nrows, ncols=1, height_ratios=height_ratios)

    # --- accuracy axis ---
    ax_acc = fig.add_subplot(gs[0, 0])
    x = np.arange(n_epochs)
    ax_acc.plot(x, np.asarray(accs, dtype=float), linewidth=1.5)
    ax_acc.set_xlim(-0.5, n_epochs - 0.5)
    ax_acc.set_ylabel(acc_ylabel)
    if acc_ylim is not None:
        ax_acc.set_ylim(*acc_ylim)
    ax_acc.grid(True, axis="y", alpha=0.3)
    ax_acc.tick_params(axis="x", labelbottom=False)

    # --- heatmap axes ---
    ax_top = fig.add_subplot(gs[1, 0])
    if type(matrix_mid) == type(None):
        ax_bottom = fig.add_subplot(gs[2, 0], sharex=ax_top, sharey=ax_top)
        heat_axes = [ax_top, ax_bottom]
    else:
        ax_mid = fig.add_subplot(gs[2, 0], sharex=ax_top, sharey=ax_top)
        ax_bottom = fig.add_subplot(gs[3, 0], sharex=ax_top, sharey=ax_top)
        heat_axes = [ax_top, ax_mid, ax_bottom]

    # use one common color range
    global_vmin = 0 #min(min(m.min(), 0.0) for m in matrices)
    global_vmax = 1 #max(_second_highest_value(m) for m in matrices)

    last_im = None
    for ax, mat, title in zip(heat_axes, matrices, titles[1:]):
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            vmin=global_vmin,
            vmax=global_vmax,
        )

        last_im = im

        ax.set_title(title)
        ax.set_ylabel("Layer")
        ax.set_xlim(-0.5, n_epochs - 0.5)
        ax.set_ylim(n_layers - 0.5, -0.5)

        # sparse ticks only
        xstep = max(1, n_epochs // 20)
        ystep = max(1, n_layers // 12)

        xticks = np.arange(0, n_epochs, xstep)
        yticks = np.arange(0, n_layers, ystep)

        ax.set_xticks(xticks)
        ax.set_xticklabels([str(epochs[i]) for i in xticks], rotation=90)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(layers[i]) for i in yticks])

    ax_bottom.set_xlabel("Epoch")

    # one shared colorbar for all heatmaps
    if last_im is not None:
        fig.colorbar(last_im, ax=heat_axes, shrink=0.9, pad=0.01)

    # --- images strip ---
    if want_images:
        n_imgs = len(img_indices)
        rel_paths, labels = _read_imglist(imglist_pth)

        sub = gs[len(matrices) + 1, 0].subgridspec(nrows=1, ncols=n_imgs, wspace=0.02)
        for k, idx in enumerate(img_indices):
            ax_img = fig.add_subplot(sub[0, k])
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.axis("off")

            if not (0 <= idx < len(rel_paths)):
                continue

            img_path = _resolve_image_path(data_dir, rel_paths[idx])
            try:
                im = _load_preview_image(img_path, max_side=96)
                ax_img.imshow(im, interpolation="nearest")
                ax_img.text(
                0.02,
                0.04,
                f"{acts_values[k]:.2f}",
                transform=ax_img.transAxes,
                fontsize=7,
                color="white",
                ha="left",
                va="bottom",
                bbox=dict(
                    facecolor="black",
                    alpha=0.65,
                    pad=1.2,
                    edgecolor="none",
                ),
                )
                if show_image_labels:
                    ax_img.set_title(f"i={idx}, y={labels[idx]}", fontsize=6, pad=2)
            except Exception:
                pass

        ax_cat = fig.add_subplot(gs[len(matrices)+2, 0])
        ax_cat.set_xlim(-0.5, n_imgs - 0.5)
        ax_cat.set_ylim(0, 1)
        ax_cat.set_yticks([])

        # Example grouping into 3 equal groups if divisible by 3
        if n_imgs % 3 == 0:
            group_size = n_imgs // 3
            centers = [group_size * i + (group_size - 1) / 2 for i in range(3)]
            ax_cat.set_xticks(centers)
            ax_cat.set_xticklabels(["low", "mid", "high"])

            for cut in range(group_size, n_imgs, group_size):
                ax_cat.axvline(cut - 0.5, ymin=0.0, ymax=1.0, alpha=0.35)

        for spine in ("top", "left", "right"):
            ax_cat.spines[spine].set_visible(False)
        ax_cat.tick_params(axis="x", length=0)
        ax_cat.set_xlabel("Category")

    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _save_figure_light(fig: plt.Figure, save_path: Path, dpi: int = 150) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = save_path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        fig.savefig(save_path, dpi=dpi, quality=85, bbox_inches="tight")
    else:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

def _default_sae_encode_feature(
    sae: Any,
    x_tokens: torch.Tensor,   # [M, D]
    feature_idx: int,
) -> torch.Tensor:
    """
    Default assumption:
      - sae.encode(x_tokens) returns [M, F]
      or
      - sae(x_tokens) returns [M, F]
    Adapt this if your SAE API differs.
    """
    if hasattr(sae, "encode"):
        z = sae.encode(x_tokens)
    else:
        z = sae(x_tokens)

    if isinstance(z, tuple):
        z = z[0]

    if z.ndim != 2:
        raise ValueError(
            f"Expected SAE output shape [num_tokens, num_features], got {tuple(z.shape)}"
        )

    return z[:, feature_idx]   # [M]

def _compute_patch_ranking_scores(
    *,
    patch_latents: Optional[Sequence[np.ndarray]] = None,   # raw [T,D]
    patch_activations: Optional[Sequence[np.ndarray]] = None,  # precomputed [T], [H,W], or [1+T]
    sae: Optional[Any] = None,
    feature_idx: Optional[int] = None,
    patch_grid: Optional[Tuple[int, int]] = None,
    drop_cls_for_patch_ranking: bool = True,
    patch_rank_reduce: Literal["max", "mean", "sum_positive", "q95"] = "max",
    device: str = "cpu",
    sae_token_batch_size: int = 4096,
    sae_encode_feature_fn: Optional[Callable[[Any, torch.Tensor, int], torch.Tensor]] = None,
) -> np.ndarray:
    """
    Returns one scalar patch-based ranking score per image.

    Memory-efficient:
      - does NOT store patch maps for all images
      - only stores one scalar per image
    """
    if sae_encode_feature_fn is None:
        sae_encode_feature_fn = _default_sae_encode_feature

    # Case 1: patch activations are already given
    if patch_activations is not None:
        scores = np.empty(len(patch_activations), dtype=float)

        for i, pa in enumerate(patch_activations):
            pm = _tokens_to_patch_map(
                np.asarray(pa),
                patch_grid=patch_grid,
                drop_cls_token=drop_cls_for_patch_ranking,
            )
            scores[i] = _reduce_token_scores(pm, mode=patch_rank_reduce)

        return scores

    # Case 2: compute patch feature activations from raw patch latents through SAE
    if patch_latents is None or sae is None or feature_idx is None:
        raise ValueError(
            "For patch-based ranking, provide either `patch_activations` "
            "or (`patch_latents`, `sae`, and `feature_idx`)."
        )

    scores = np.empty(len(patch_latents), dtype=float)

    for i, toks in enumerate(patch_latents):
        if isinstance(toks, torch.Tensor):
            toks_np = toks.detach().cpu().numpy()
        else:
            toks_np = np.asarray(toks)

        if toks_np.ndim != 2:
            raise ValueError(
                f"patch_latents[{i}] must have shape [T,D] for patch ranking, got {toks_np.shape}"
            )

        toks_np_spatial = toks_np[1:] if drop_cls_for_patch_ranking else toks_np
        num_tokens = toks_np_spatial.shape[0]

        token_scores_chunks = []
        for start in range(0, num_tokens, sae_token_batch_size):
            chunk = toks_np_spatial[start:start + sae_token_batch_size]
            chunk_t = torch.as_tensor(chunk, dtype=torch.float32, device=device)

            with torch.no_grad():
                feat_scores = sae_encode_feature_fn(sae, chunk_t, feature_idx)

            token_scores_chunks.append(feat_scores.detach().cpu().numpy())

        token_scores = np.concatenate(token_scores_chunks, axis=0)  # [T]
        scores[i] = _reduce_token_scores(token_scores, mode=patch_rank_reduce)

    return scores


def _encode_patch_maps_for_selected(
    selected_indices: Sequence[int],
    patch_latents: Sequence[np.ndarray],
    sae: Any,
    feature_idx: int,
    *,
    patch_grid: Optional[Tuple[int, int]] = None,
    drop_cls_for_heatmap: bool = False,
    device: str = "cpu",
    sae_token_batch_size: int = 4096,
    sae_encode_feature_fn: Optional[Callable[[Any, torch.Tensor, int], torch.Tensor]] = None,
) -> Dict[int, np.ndarray]:
    """
    Encodes patch tokens through the SAE only for selected images.

    Returns:
      dict: image_index -> patch_map [H, W]
    """
    if sae_encode_feature_fn is None:
        sae_encode_feature_fn = _default_sae_encode_feature

    patch_map_dict: Dict[int, np.ndarray] = {}

    for i in selected_indices:
        toks = patch_latents[i]
        if isinstance(toks, torch.Tensor):
            toks_np = toks.detach().cpu().numpy()
        else:
            toks_np = np.asarray(toks)

        if toks_np.ndim != 2:
            raise ValueError(
                f"patch_latents[{i}] must have shape [T,D], got {toks_np.shape}"
            )

        # split off CLS for spatial map
        if drop_cls_for_heatmap:
            toks_np_spatial = toks_np[1:] if toks_np.shape[0] >= 2 else toks_np
        else:
            toks_np_spatial = toks_np

        num_tokens = toks_np_spatial.shape[0]

        if patch_grid is None:
            side = int(round(np.sqrt(num_tokens)))
            if side * side != num_tokens:
                raise ValueError(
                    f"Cannot infer square grid from {num_tokens} patch tokens for image {i}. "
                    f"Pass patch_grid explicitly."
                )
            gh, gw = side, side
        else:
            gh, gw = patch_grid
            if gh * gw != num_tokens:
                raise ValueError(
                    f"patch_grid={patch_grid} incompatible with {num_tokens} spatial tokens "
                    f"for image {i}."
                )

        token_scores_chunks = []
        for start in range(0, num_tokens, sae_token_batch_size):
            chunk = toks_np_spatial[start:start + sae_token_batch_size]
            chunk_t = torch.as_tensor(chunk, dtype=torch.float32, device=device)

            with torch.no_grad():
                feat_scores = sae_encode_feature_fn(sae, chunk_t, feature_idx)

            token_scores_chunks.append(feat_scores.detach().cpu().numpy())

        token_scores = np.concatenate(token_scores_chunks, axis=0)  # [T]
        patch_map = token_scores.reshape(gh, gw)
        patch_map_dict[i] = patch_map

    return patch_map_dict
def plot_top_activating_images_per_class(
    num_classes: int,
    activations: Optional[Sequence[float]],   # CLS-based scores if ranking_source="cls"
    save_path: Path,
    *,
    # Option A: provide labels + image paths directly
    labels: Optional[Sequence[int]] = None,
    img_paths: Optional[Sequence[str]] = None,
    # Option B: provide dataset roots like before
    data_dir: Optional[Path] = None,
    imglist_pth: Optional[Path] = None,
    # Ranking mode
    ranking_source: Literal["cls", "patch"] = "cls",
    patch_rank_reduce: Literal["max", "mean", "sum_positive", "q95"] = "max",
    drop_cls_for_patch_ranking: bool = False,
    # Memory-efficient patch-SAE support
    sae: Optional[Any] = None,
    patch_latents: Optional[Sequence[np.ndarray]] = None,  # raw token embeddings [T,D] or [1+T,D]
    feature_idx: Optional[int] = None,
    patch_grid: Optional[Tuple[int, int]] = None,
    drop_cls_for_heatmap: bool = False,
    sae_device: str = "cpu",
    sae_token_batch_size: int = 4096,
    sae_encode_feature_fn: Optional[Callable[[Any, torch.Tensor, int], torch.Tensor]] = None,
    # Or pass precomputed patch activations
    patch_activations: Optional[Sequence[np.ndarray]] = None,
    heatmap_alpha: float = 0.38,
    heatmap_cmap: str = "jet",
    heatmap_positive_only: bool = True,
    # plotting options
    top_k: int = 10,
    preview_max_side: int = 96,
    title: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    sort_descending: bool = True,
    show_only_positive_summary: bool = True,
    max_plotted_classes: int = 20,
) -> None:
    """
    Memory-efficient version:
      - rank by CLS scores or patch scores
      - patch heatmaps are computed only for selected images

    ranking_source:
      - "cls": use `activations`
      - "patch": derive one scalar score per image from patch activations

    If num_classes > max_plotted_classes, only classes with the highest
    single-image positive activation are plotted.

    This means that a class with only one strongly activating image can be shown
    instead of a large class whose images are all weakly activated.
    """

    # --- load image paths / labels ---
    if labels is not None and img_paths is not None:
        labels_arr = np.asarray(labels, dtype=int)
        resolved_paths = [Path(p) for p in img_paths]

    elif data_dir is not None and imglist_pth is not None:
        rel_paths, lbls = _read_imglist(imglist_pth)
        labels_arr = np.asarray(lbls, dtype=int)
        resolved_paths = [_resolve_image_path(data_dir, rp) for rp in rel_paths]

    else:
        raise ValueError(
            "Provide either (labels and img_paths) or (data_dir and imglist_pth)."
        )

    n = len(labels_arr)

    if class_names is not None and len(class_names) != num_classes:
        raise ValueError(
            f"len(class_names)={len(class_names)} must match num_classes={num_classes}"
        )

    # --- prepare ranking scores ---
    cls_scores = None

    if activations is not None:
        cls_scores = np.asarray(activations, dtype=float)

        if len(cls_scores) != n:
            raise ValueError(
                f"Length mismatch: len(activations)={len(cls_scores)}, len(labels)={n}"
            )

    if ranking_source == "cls":
        if cls_scores is None:
            raise ValueError("`activations` must be provided when ranking_source='cls'.")

        ranking_scores = cls_scores.copy()

    elif ranking_source == "patch":
        ranking_scores = _compute_patch_ranking_scores(
            patch_latents=patch_latents,
            patch_activations=patch_activations,
            sae=sae,
            feature_idx=feature_idx,
            patch_grid=patch_grid,
            drop_cls_for_patch_ranking=drop_cls_for_patch_ranking,
            patch_rank_reduce=patch_rank_reduce,
            device=sae_device,
            sae_token_batch_size=sae_token_batch_size,
            sae_encode_feature_fn=sae_encode_feature_fn,
        )

        if len(ranking_scores) != n:
            raise ValueError(
                f"Length mismatch: len(ranking_scores)={len(ranking_scores)}, len(labels)={n}"
            )

    else:
        raise ValueError(f"Unknown ranking_source: {ranking_source}")

    # --- validity mask ---
    valid = np.isfinite(ranking_scores)

    ranking_scores = ranking_scores[valid]
    labels_arr = labels_arr[valid]
    resolved_paths = [p for p, v in zip(resolved_paths, valid) if v]

    if cls_scores is not None:
        cls_scores = cls_scores[valid]

    if patch_latents is not None:
        patch_latents = [pl for pl, v in zip(patch_latents, valid) if v]

    if patch_activations is not None:
        patch_activations = [pa for pa, v in zip(patch_activations, valid) if v]

    # --- per-class ordering using chosen ranking scores ---
    class_to_sorted_indices: List[np.ndarray] = []

    # Used only for selecting which classes to show when there are too many.
    # For each class, this stores the maximum positive activation in that class.
    class_peak_activation = np.full(num_classes, -np.inf, dtype=float)

    for c in range(num_classes):
        idx = np.where(labels_arr == c)[0]

        # Keep only positive scores.
        idx = idx[ranking_scores[idx] > 0.0]

        if idx.size == 0:
            class_to_sorted_indices.append(idx)
            continue

        class_peak_activation[c] = float(np.max(ranking_scores[idx]))

        order = np.argsort(ranking_scores[idx])

        if sort_descending:
            order = order[::-1]

        class_to_sorted_indices.append(idx[order])

    # --- choose classes to plot ---
    # If there are more than max_plotted_classes, select classes by their
    # strongest single activating image, not by how many images the class has.
    valid_classes = np.where(np.isfinite(class_peak_activation))[0]

    if len(valid_classes) == 0:
        raise ValueError("No classes with positive valid activations to plot.")

    if num_classes > max_plotted_classes:
        class_order = valid_classes[
            np.argsort(class_peak_activation[valid_classes])[::-1]
        ]
        plotted_classes = class_order[:max_plotted_classes]
    else:
        plotted_classes = valid_classes

    n_plot_classes = len(plotted_classes)

    # --- identify only the images that will be shown ---
    selected_indices: List[int] = []

    for c in plotted_classes:
        sorted_idx = class_to_sorted_indices[c]
        n_show = min(top_k, len(sorted_idx))
        selected_indices.extend(sorted_idx[:n_show].tolist())

    selected_indices = sorted(set(selected_indices))

    # --- build patch maps only for selected images ---
    patch_map_dict: Dict[int, np.ndarray] = {}

    if patch_activations is not None:
        for i in selected_indices:
            pm = _tokens_to_patch_map(
                np.asarray(patch_activations[i]),
                patch_grid=patch_grid,
                drop_cls_token=drop_cls_for_heatmap,
            )
            patch_map_dict[i] = pm

    elif sae is not None and patch_latents is not None and feature_idx is not None:
        patch_map_dict = _encode_patch_maps_for_selected(
            selected_indices=selected_indices,
            patch_latents=patch_latents,
            sae=sae,
            feature_idx=feature_idx,
            patch_grid=patch_grid,
            drop_cls_for_heatmap=drop_cls_for_heatmap,
            device=sae_device,
            sae_token_batch_size=sae_token_batch_size,
            sae_encode_feature_fn=sae_encode_feature_fn,
        )

    # --- top summary: normalized activation/ranking mass by class ---
    if show_only_positive_summary:
        weights = np.clip(ranking_scores, 0.0, None)

        if weights.sum() == 0:
            weights = np.abs(ranking_scores)
    else:
        weights = np.abs(ranking_scores)

    class_mass = np.bincount(
        labels_arr,
        weights=weights,
        minlength=num_classes,
    ).astype(float)

    if class_mass.sum() > 0:
        class_mass /= class_mass.sum()

    class_mass_plot = class_mass[plotted_classes]

    # --- figure layout ---
    fig_w = max(14.0, 1.45 * top_k)
    fig_h = max(2.8 + 1.45 * n_plot_classes, 6.0)

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    gs = fig.add_gridspec(
        nrows=n_plot_classes + 1,
        ncols=top_k,
        height_ratios=[1.3] + [1.0] * n_plot_classes,
        hspace=0.04,
        wspace=0.02,
    )

    # --- top bar plot ---
    ax_top = fig.add_subplot(gs[0, :])

    x = np.arange(n_plot_classes)

    ax_top.bar(x, class_mass_plot)
    ax_top.set_xlim(-0.5, n_plot_classes - 0.5)
    ax_top.set_ylabel("Norm. mass")

    if class_names is None:
        ax_top.set_xticks(x)
        ax_top.set_xticklabels([str(c) for c in plotted_classes])
    else:
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(
            [class_names[c] for c in plotted_classes],
            rotation=45,
            ha="right",
        )

    if title is not None:
        ax_top.set_title(title)
    else:
        if num_classes > max_plotted_classes:
            ax_top.set_title(
                f"Normalized activation mass by class "
                f"({ranking_source} ranking, top {n_plot_classes} classes by peak activation)"
            )
        else:
            ax_top.set_title(
                f"Normalized activation mass by class ({ranking_source} ranking)"
            )

    ax_top.grid(True, axis="y", alpha=0.25)

    # --- image grid ---
    for row_idx, c in enumerate(plotted_classes):
        sorted_idx = class_to_sorted_indices[c]
        n_show = min(top_k, len(sorted_idx))

        for j in range(top_k):
            ax = fig.add_subplot(gs[row_idx + 1, j])
            ax.set_xticks([])
            ax.set_yticks([])

            if j >= n_show:
                ax.set_axis_off()
                continue

            i = sorted_idx[j]
            img_path = resolved_paths[i]
            score = float(ranking_scores[i])

            try:
                im = _load_preview_image(img_path, max_side=preview_max_side)
                im = np.asarray(im)
                h_img, w_img = im.shape[:2]
                ax.imshow(im, interpolation="nearest")

            except Exception:
                ax.set_axis_off()
                continue

            if i in patch_map_dict:
                pm = _normalize_patch_map_for_display(
                    patch_map_dict[i],
                    positive_only=heatmap_positive_only,
                )

                ax.imshow(
                    pm,
                    cmap=heatmap_cmap,
                    alpha=heatmap_alpha,
                    interpolation="bilinear",
                    extent=(0, w_img, h_img, 0),
                )

            ax.text(
                0.02,
                0.04,
                f"{score:.2f}",
                transform=ax.transAxes,
                fontsize=8,
                color="white",
                ha="left",
                va="bottom",
                bbox=dict(
                    facecolor="black",
                    alpha=0.65,
                    pad=1.2,
                    edgecolor="none",
                ),
            )

            if j == 0:
                row_label = class_names[c] if class_names is not None else f"class {c}"

                ax.set_ylabel(
                    row_label,
                    rotation=0,
                    labelpad=28,
                    va="center",
                    fontsize=12,
                )

    _save_figure_light(fig, save_path, dpi=300)
    plt.close(fig)

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
            elif method == "hungarian":
                cos_res[:,li,ei] = align_concepts_hungarian(sim_res_dict["sim_cos"].T,)["match_scores"]
                cor_res[:,li,ei] = align_concepts_hungarian(sim_res_dict["sim_cor"].T,)["match_scores"]
                top_res[:,li,ei] = align_concepts_hungarian(sim_res_dict["sim_topk"].T,)["match_scores"]

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


    picked_concepts = query_concepts(concepts_stats,
                        bounds=criteria,
                        sort_key="label_entropy",  # any metric in the dict
                        ascending=True,            # smallest entropy first
                        return_scores=False)        # → [(id, metric_dict), …]
    
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
    #_plot_heatmap(
    #    mean_align,
    #    epochs,
    #    layers,
    #    f"Align Hungarian ({args['align_summary_key']}) - mean over runs_ref: epoch {ref_epoch} layer{ref_layer}",
    #    plots_dir / f"align_{args['align_summary_key']}_mean_ref_e{ref_epoch}_l{ref_layer}.png")

    _plot_heatmap(
        mean_topk,
        epochs,
        layers,
        f"TopK Sets ({args['topk_summary_key']}) - mean over runs_ref: epoch {ref_epoch} layer{ref_layer}",
        plots_dir / f"topk_{args['topk_summary_key']}_mean_ref_e{ref_epoch}_l{ref_layer}.png",
    )
import numpy as np

ref_epochs = [0, 25, 49, 50, 51, 100, 150, 200, 235, 245, 249, 250, 251, 299]

half_window = 50      # because total window is 100
n_epochs = 300        # epochs are assumed to be 0, ..., 299
low_stride = 10

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
    if eval_device != "cpu":
        if args_dict["train_saes"]:
            _train_all_saes(args_dict, epochs, layers,ref_epoch, ref_layer, cache)

    #all_align = np.full((args_dict["sae_runs"], len(layers), len(epochs)), np.nan, dtype=float)
    all_topk = np.full((args_dict["sae_runs"], len(layers), len(epochs)), np.nan, dtype=float)



    print(eval_device)
    for run_idx in range(args_dict["sae_runs"]):
        
        if args_dict["ind_name"] == "shapes3d":
            ref_ckpt = _checkpoint_path(args_dict, ref_epoch, ref_layer, run_idx)
            if not ref_ckpt.exists():
                raise FileNotFoundError(f"Missing reference SAE checkpoint: {ref_ckpt}")

            ref_sae = _load_sae_for_eval(ref_ckpt, eval_device)
            ref_data = _load_all_data_cached(ref_epoch, ref_layer, args_dict, cache)
            heatmap_dir = plots_dir /f"run_{run_idx}"
            _ensure_dir(heatmap_dir)
            make_heatmap(ref_sae,ref_data, args_dict, 1e-3, args_dict["eval_split"], args_dict["backbone"], heatmap_dir , eval_device)
        
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
