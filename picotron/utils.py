import json
import os
import subprocess
import requests
import random
import numpy as np
import builtins
import fcntl

from safetensors import safe_open

import picotron.process_group_manager as pgm
import torch, torch.distributed as dist

def print(*args, is_print_rank=True, **kwargs):
    """ solves multi-process interleaved print problem """
    if not is_print_rank: return
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
def to_readable_format(num, precision=2):
    if num >= 1e12:
        return f"{num / 1e12:.{precision}f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

# ref: 
# https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L289
# https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_02.py#L950
def get_mfu(tokens_per_second, num_params, model_config, theoretical_flops = 989.5 * 10 ** 12):
    num_layers = model_config.num_hidden_layers
    hidden_dim = model_config.hidden_size
    seq_len = model_config.max_position_embeddings
    flops_per_token = 6 * num_params + 12 * num_layers * hidden_dim * seq_len
    mfu = tokens_per_second * flops_per_token / theoretical_flops * 100 # percentage
    return mfu

def get_num_params(model):
    """Calculate total number of parameters accounting for tensor parallelism and pipeline parallelism.
    
    For TP: Parameters in attention/mlp/embed/final_proj are sharded, so multiply by tp_world_size
    For PP: Need to gather parameter counts across pipeline stages
    For DP: Parameters are replicated, so only count once
    
    Note: 
    FSDP: Parameters are sharded across data parallel ranks
    """
    tp_world_size = pgm.process_group_manager.tp_world_size
    
    # Count parameters in current PP rank
    local_num_params = 0
    for name, param in model.named_parameters():
        # Parameters split across TP ranks
        # TODO: LayerNorm is also split across TP ranks for sequence parallelism
        if any(tp_keyword in name.lower() for tp_keyword in ['attention', 'mlp', 'embed', 'final_proj']):
            local_num_params += param.numel() * tp_world_size
        else:
            # Parameters replicated across TP ranks (layer norm, biases)
            local_num_params += param.numel()
            
    # Gather parameter counts from all PP ranks
    param_counts = torch.tensor(local_num_params, device='cuda')
    
    # Sum up parameters across all PP ranks
    dist.all_reduce(param_counts, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.pp_group)
    
    return param_counts.item()
    
def assert_no_meta_tensors(model):
    meta_tensors = []
    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            meta_tensors.append(f"Parameter '{name}' with shape {param.shape}")
    
    for name, buffer in model.named_buffers():
        if buffer.device == torch.device("meta"):
            meta_tensors.append(f"Buffer '{name}' with shape {buffer.shape}")
    
    assert len(meta_tensors) == 0, f"Found {len(meta_tensors)} meta tensors:\n" + "\n".join(meta_tensors)

def average_loss_across_dp_cp_ranks(loss, device):
    reduced_loss = torch.tensor([loss if loss is not None else 0.0], dtype=torch.float32, device=device)
    if pgm.process_group_manager.pp_is_last_stage:
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
        reduced_loss /= pgm.process_group_manager.cp_dp_world_size
    return reduced_loss.item()

def check_hf_model_files_existences(model_name, hf_token):
    files_to_check = [
        "model.safetensors",
        "model.safetensors.index.json"
    ]

    # Prepare headers with authentication token
    headers = {}
    if hf_token: headers["Authorization"] = f"Bearer {hf_token}"

    index = 0
    found_files = []
    for file in files_to_check:
        url = f'https://huggingface.co/{model_name}/resolve/main/{file}'
        try:
            # Use GET request with stream=True and authentication headers
            response = requests.get(url, stream=True, headers=headers)
            if response.status_code == 200:
                found_files.append(file)
                print(f"✅ Found {file}")
                response.close()
            elif response.status_code == 401:
                print(f"❌ Authentication required for {file} (Status: {response.status_code})")
            elif response.status_code == 403:
                print(f"❌ Access denied for {file} (Status: {response.status_code})")
            else:
                print(f"❌ Not found {file} (Status: {response.status_code})")
        except Exception as e:
            print(f"❌ Error checking {file}: {str(e)}")

    return found_files

def download_hf_model_files(files_to_download, model_name, hf_token, save_dir):
    downloaded_files = []

    save_dir_path = f"{save_dir}/{model_name}"

    for file in files_to_download:
        if os.path.exists(os.path.join(save_dir_path, file)):
            print(f"✅ {file} already exists")
            downloaded_files.append(file)

            # If it's index.json, read it to get shards
            if file.endswith('.json'):
                with open(os.path.join(save_dir_path, file), 'r') as f:
                    index_data = json.load(f)
                    shards = set(index_data['weight_map'].values())
                    print(f"Found {len(shards)} shards in index")
                    files_to_download.extend(shards)
            continue

        model_cmd = f"huggingface-cli download {model_name} {file} --local-dir {save_dir_path} --token {hf_token}"
        print(f"Downloading {file}...")
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(model_cmd, shell=True, check=False, env=env)

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        if result.returncode == 0:
            print(f"✅ {file} downloaded successfully")
            downloaded_files.append(file)

            # Verify files based on their type
            file_path = os.path.join(save_dir_path, file)
            if file.endswith('.safetensors'):
                try:
                    with safe_open(file_path, framework="pytorch", device="cpu") as f:
                        keys = list(f.keys())
                        print(f"✅ Safetensors file is valid")
                        print(f"- Number of tensors: {len(keys)}")
                except Exception as e:
                    print(f"❌ Error validating safetensors file: {str(e)}")
                    continue
            elif file.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        index_data = json.load(f)
                        shards = set(index_data['weight_map'].values())
                        print(f"✅ Index JSON file is valid")
                        print(f"- Number of weight shards: {len(shards)}")
                        # Add shards to files_to_download
                        files_to_download.extend(shards)
                except Exception as e:
                    print(f"❌ Error validating index JSON file: {str(e)}")
                    continue
        else:
            error_message = result.stderr.decode('utf-8', errors='replace')
            if "404 Client Error" in error_message or "Entry Not Found" in error_message:
                print(f"❌ File {file} not found in repository")
            else:
                print(f"❌ Download failed: {error_message.strip()}")

    print(f"\nSuccessfully downloaded files: {', '.join(downloaded_files)}")
    return True

def download_model(model_name, hf_token):
    # Download HF model safetensors at the "hf_model_safetensors" directory
    os.makedirs("hf_model_safetensors", exist_ok=True)

    files_to_download = check_hf_model_files_existences(model_name, hf_token)
    if len(files_to_download) <= 0:
        raise FileNotFoundError("Safetensors files not found. Please check the model name and authentication token.")

    is_downloaded = download_hf_model_files(files_to_download, model_name, hf_token, save_dir="hf_model_safetensors")
    if not is_downloaded:
        raise FileNotFoundError("Failed to download safetensors files. Please check the model name and authentication token.")

    print("SafeTensors files downloaded successfully! ✅")