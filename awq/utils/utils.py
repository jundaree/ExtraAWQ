import torch
import accelerate
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


def simple_dispatch_model(model, device_map):
    from accelerate.hooks import add_hook_to_module, AlignDevicesHook

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {
        "cpu",
        "disk",
    }:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(
            m, execution_device=main_device, prev_module_hook=prev_hook
        )
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(
            model, cpu_offload_group[0][0]
        )._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model


def save_extra_scales(scales_list, save_path=None): 
    

    # Create lists to store all values from s[2] and s[3]
    all_s2 = []
    all_s3 = []
    
    # Extract values from each tuple in scales_list
    for s in scales_list:
        if isinstance(s[2], torch.Tensor) and isinstance(s[3], torch.Tensor):
            # Convert tensors to numpy arrays and flatten
            s2_values = s[2].flatten().cpu().numpy()
            s3_values = s[3].flatten().cpu().numpy()
            
            # Ensure both arrays have the same length
            min_length = min(len(s2_values), len(s3_values))
            all_s2.extend(s2_values[:min_length])
            all_s3.extend(s3_values[:min_length])
    
    # Convert lists to numpy arrays
    all_s2_array = np.array(all_s2)
    all_s3_array = np.array(all_s3)
    
    # Save both arrays in a single pth file
    Path(save_path).mkdir(exist_ok=True)
    torch.save({
        "org_scales": all_s3_array,
        "scales": all_s2_array
    }, Path(save_path)/"scales_data.pth")
    

    