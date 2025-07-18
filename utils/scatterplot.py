# Create the scatter plot
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import torch


data_path = Path('/workspace/ExtraAWQ/awq_cache')

argparser = argparse.ArgumentParser(description="Generate line plots for model scales")
argparser.add_argument("--llm", type=str, required=True, help="name of the LLM (e.g., 'opt', 'llama', 'Llama-2')")
args = argparser.parse_args()
llm= args.llm
if llm == "opt":
    llm_title = "OPT"
elif llm == "llama":
    llm_title = "LLaMA"
elif llm == "Llama-2":
    llm_title = llm
else:
    raise ValueError(f"Unsupported LLM name: {llm}. Supported names are 'opt', 'llama', 'Llama-2'.")




def extract_model_size(folder_name):
    """Extract the model size in billions from the folder name."""
    # Extract the part between "opt-" and "b-"
    try:
        size_str = folder_name.split('opt-')[1].split('b-')[0]
        # Convert to float (handles cases like 1.3, 2.7, etc.)
        return float(size_str)
    except (IndexError, ValueError):
        # Default value for folders that don't match the pattern
        return float('inf')  # Put them at the end



org_scales = {}
scales = {}

# Get all matching folders
matching_folders = list(data_path.glob(f"{llm}*"))
# Sort folders by model size
sorted_folders = sorted(matching_folders, key=lambda x: extract_model_size(x.name))

for folder in sorted_folders:
    scales_file = folder / "scales_data.pth"
    if scales_file.exists():
        s= torch.load(scales_file)
        org_scales[folder.name] =s['org_scales']
        scales[folder.name] = s['scales']

# Print the number of items found to debug
print(f"Found {len(org_scales)} items for {llm}")

plt.figure(figsize=(10, 8))

# Create a colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(org_scales)))

# Check if we have any data
if len(org_scales) == 0:
    print("No data found. Please check if there are any matching folders.")
    exit(1)

for i, key in enumerate(org_scales.keys()):
    # Use a more descriptive label by removing the common prefix
    label = key.replace(f"{llm}-", "")
    # Ensure label is not empty or starting with underscore
    if not label or label.startswith('-'):
        label = f"model_{i+1}"
    
    # Print the key and processed label for debugging
    print(f"Processing key: {key}, label: {label}")
    print(f"length of org_scales: {len(org_scales[key])}, length of scales: {len(scales[key])}")
    
    # Make sure to actually use the label in the scatter plot
    scatter = plt.scatter(org_scales[key], scales[key], alpha=0.6, s=100, color=colors[i], label=label)

# Increase font size for title and labels
plt.title(f'{llm_title}', fontsize=35)
plt.xlabel('scales before ExtraAWQ', fontsize=24)
plt.ylabel('scales after ExtraAWQ', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend inside the plot with larger font
plt.legend(loc='upper center', fontsize=25, markerscale=2.0)

# Increase tick label font sizes
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.tight_layout()
# plt.show()
plt.savefig(data_path / f"scales_scatter_plot_{llm}.png", bbox_inches='tight')
