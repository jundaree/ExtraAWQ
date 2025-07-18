# Create the scatter plot
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import torch
import argparse

data_path = Path('/workspace/ExtraAWQ/awq_cache')

argparser = argparse.ArgumentParser(description="Generate line plots for model scales")
argparser.add_argument("--llm", type=str, required=True, help="name of the LLM (e.g., 'opt', 'llama', 'Llama-2')")
args = argparser.parse_args()
llm = args.llm
if llm == "opt":
    llm_title = "OPT"
elif llm == "llama":
    llm_title = "LLaMA"
elif llm == "Llama-2":
    llm_title = llm
else:
    raise ValueError(f"Unsupported LLM name: {llm}. Supported names are 'opt', 'llama', 'Llama-2'.")
default=llm
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

# Function to create adaptive bins and calculate average y values
def create_binned_data(x, y, n_bins=50):
    """Create adaptive bins for x and calculate average y for each bin."""
    # Sort the data by x values
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Calculate bin edges to have roughly equal number of points in each bin
    n_points = len(x)
    points_per_bin = n_points // n_bins
    
    bin_x = []
    bin_y = []
    
    for i in range(0, n_points, points_per_bin):
        end_idx = min(i + points_per_bin, n_points)
        if end_idx <= i:
            continue
            
        bin_x_value = np.mean(x_sorted[i:end_idx])
        bin_y_value = np.mean(y_sorted[i:end_idx])
        
        bin_x.append(bin_x_value)
        bin_y.append(bin_y_value)
    
    print(f"Created {len(bin_x)} bins with average y values.")

    return np.array(bin_x), np.array(bin_y)

# Function to create fixed-width bins and calculate average y values
def create_fixed_binned_data(x, y, bin_width=0.05, x_min=0, x_max=3):
    """Create fixed-width bins for x and calculate average y for each bin."""
    # Create bin edges with fixed width
    bin_edges = np.arange(x_min, x_max + bin_width, bin_width)
    n_bins = len(bin_edges) - 1
    
    bin_x = []
    bin_y = []
    
    for i in range(n_bins):
        # Get points that fall within this bin
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if np.sum(mask) > 0:  # Only include bins that have data points
            bin_x_value = (bin_edges[i] + bin_edges[i+1]) / 2  # Center of the bin
            bin_y_value = np.mean(y[mask])
            
            bin_x.append(bin_x_value)
            bin_y.append(bin_y_value)
    
    print(f"Created {len(bin_x)} fixed-width bins with average y values.")
    return np.array(bin_x), np.array(bin_y)

# Full range plot (first plot)
plt.figure(figsize=(10, 8))

# Create a colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(org_scales)))

# Set range and bin width for the first plot
full_range_min = 0
full_range_max = np.max([np.max(org_scales[key]) for key in org_scales.keys()]) * 1.1
full_bin_width = 0.5

for i, key in enumerate(org_scales.keys()):
    # Use a more descriptive label by removing the common prefix
    label = key.replace(f"{llm}-", "")
    # Ensure label is not empty or starting with underscore
    if not label or label.startswith('_'):
        label = f"model_{i+1}"
    
    # Print the key and processed label for debugging
    print(f"Processing key: {key}, label: {label}")
    print(f"length of org_scales: {len(org_scales[key])}, length of scales: {len(scales[key])}")
    
    # Convert to numpy arrays for binning
    x = np.array(org_scales[key])
    y = np.array(scales[key])
    
    # Filter data for the full range
    mask = (x >= full_range_min) & (x <= full_range_max)
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    if len(x_filtered) > 0:
        # Use fixed-width bins for the first plot too
        bin_x, bin_y = create_fixed_binned_data(x_filtered, y_filtered, 
                                              bin_width=full_bin_width, 
                                              x_min=full_range_min, 
                                              x_max=full_range_max)
        
        # Plot the line connecting average y values for each bin
        plt.plot(bin_x, bin_y, linewidth=3, color=colors[i], label=label)

# Set x-axis and y-axis range explicitly
plt.xlim(full_range_min, full_range_max)
plt.ylim(full_range_min, full_range_max)

# Increase font size for title and labels
plt.title(f'{llm_title}', fontsize=30)
plt.xlabel('scales before ExtraAWQ', fontsize=24)
plt.ylabel('scales after ExtraAWQ', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend inside the plot with larger font
plt.legend(loc='lower right', fontsize=20)

# Increase tick label font sizes
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()
plt.savefig(data_path / f"scales_line_plot_{llm}_full.png", bbox_inches='tight')

# Create an additional plot with x-range limited to 0-2
plt.figure(figsize=(10, 8))

# Create a colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(org_scales)))

range_min = 0
range_max= 10
bin_width = 0.1

for i, key in enumerate(org_scales.keys()):
    # Use a more descriptive label by removing the common prefix
    label = key.replace(f"{llm}-", "")
    # Ensure label is not empty or starting with underscore
    if not label or label.startswith('_'):
        label = f"model_{i+1}"
    
    # Convert to numpy arrays for binning
    x = np.array(org_scales[key])
    y = np.array(scales[key])
    
    # Filter data to include only x values between 0 and 2
    mask = (x >= range_min) & (x <= range_max)
    x_filtered = x[mask]
    y_filtered = y[mask]

    print(f"length of filtered org_scales: {len(x_filtered)}, length of filtered scales: {len(y_filtered)}")
    
    if len(x_filtered) > 0:
        # Use fixed-width bins for smoother curve
        bin_x, bin_y = create_fixed_binned_data(x_filtered, y_filtered, bin_width=bin_width, x_min=range_min, x_max=range_max)
        
        # Plot the line connecting average y values for each bin
        plt.plot(bin_x, bin_y, linewidth=5, color=colors[i], label=label)

# Set x-axis range explicitly
plt.xlim(range_min, range_max)
plt.ylim(range_min, range_max)

# Increase font size for title and labels
plt.title(f'{llm_title}', fontsize=35)
plt.xlabel('scales before ExtraAWQ', fontsize=24)
plt.ylabel('scales after ExtraAWQ', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend inside the plot with larger font
plt.legend(loc='lower right', fontsize=25)

# Increase tick label font sizes
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()
plt.savefig(data_path / f"scales_line_plot_{llm}_{range_min}-{range_max}.png", bbox_inches='tight')
