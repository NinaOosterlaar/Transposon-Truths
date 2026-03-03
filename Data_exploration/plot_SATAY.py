import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# Add Signal_processing directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "Signal_processing"))
from ZINB_MLE.estimate_ZINB import estimate_zinb

CHROMS = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI"]
CPD = {
    "I": [-720, -500, -450, -200, -70, 80, 440, 517, 720, 930],
    "II": [-850, -680, -60, 65],
    "III": [-900, -200, -120, 750],
    "IV": [],
    "V": [],
    "VI": [],    
    "VII": [],
    "VIII": [],
    "IX": [],
    "X": [],
    "XI": [],
    "XII": [],
    "XIII": [],
    "XIV": [],
    "XV": [],
    "XVI": [],
}

def plot_around_centromere_x_is_centdist(
    filepath,
    outpath,
    n=1000,
    centromere_target=0,
    strict_centromere_match=True,
    major_tick_step=100,
    minor_tick_step=50,
    dpi=200,
    value_max=500,   # <-- filter threshold
    window_size=100,  # window size for ZINB parameter estimation
    show_zinb_params=True,  # whether to show ZINB parameters
):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Find centromere midpoint index
    if strict_centromere_match:
        idxs = df.index[df["centromere_distance"] == centromere_target].to_list()
        if not idxs:
            raise ValueError(
                f"{filepath}: No rows found with centromere_distance == {centromere_target}. "
                f"Set strict_centromere_match=False to use closest row instead."
            )
        cent_idx = idxs[0]
    else:
        cent_idx = (df["centromere_distance"] - centromere_target).abs().idxmin()

    # Window around centromere (±n rows)
    start = max(0, cent_idx - n)
    end = min(len(df) - 1, cent_idx + n)
    win = df.iloc[start:end + 1].copy()

    # Filter out extreme values (breaks the line at those points)
    if "value" not in win.columns:
        raise KeyError(f"{filepath}: expected a 'value' column after normalization, got {list(win.columns)}")
    win.loc[win["value"] > value_max, "value"] = float("nan")

    # Nucleosome locations in centromere_distance coords
    nuc0_x = win.loc[win["nucleosome_distance"] == 0, "centromere_distance"].to_numpy()

    # Sort by x for a clean line plot
    win = win.sort_values("centromere_distance")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(win["centromere_distance"], win["value"], linewidth=1)

    for x in nuc0_x:
        ax.axvline(x, alpha=0.35, linewidth=1)

    ax.axvline(0, linestyle="--", linewidth=2)

    # Ticks
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_step))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_step))
    ax.minorticks_on()

    # Grid (major + faint minor)
    ax.grid(True, which="major", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.5, alpha=0.15)

    ax.set_title(os.path.basename(filepath).replace("_distances.csv", ""))
    ax.set_xlabel("Centromere distance")
    ax.set_ylabel("Value")

    # Estimate ZINB parameters in windows and annotate
    if show_zinb_params:
        # Sort by position to ensure windows are sequential
        win_sorted = win.sort_values("centromere_distance").reset_index(drop=True)
        values_array = win_sorted["value"].values
        positions_array = win_sorted["centromere_distance"].values
        
        # Remove NaN values for ZINB estimation
        valid_mask = ~np.isnan(values_array)
        
        n_points = np.sum(valid_mask)
        n_windows = int(np.ceil(n_points / window_size))
        
        # Store window info for annotation
        window_annotations = []
        
        # Extract valid data for windowing
        valid_values = values_array[valid_mask]
        valid_positions = positions_array[valid_mask]
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, n_points)
            window_data = valid_values[start_idx:end_idx]
            window_positions = valid_positions[start_idx:end_idx]
            
            if len(window_data) < 10:  # Skip very small windows
                continue
            
            # Get window center position for annotation
            window_center = np.mean(window_positions)
            
            # Round to integers and filter outliers (95th percentile)
            rounded_data = np.round(window_data).astype(int)
            threshold = np.percentile(rounded_data, 95)
            filtered_data = rounded_data[rounded_data <= threshold]
            
            if len(filtered_data) < 10:  # Skip if too few points remain
                continue
            
            # Estimate ZINB parameters
            try:
                estimates = estimate_zinb(filtered_data, max_iter=1000)
                
                if estimates['converged']:
                    # Format the parameters
                    pi_val = estimates['pi']
                    mu_val = estimates['mu']
                    theta_val = estimates['theta']
                    
                    # Create annotation text
                    annot_text = f"π={pi_val:.2f}\nμ={mu_val:.1f}\nθ={theta_val:.1f}"
                    
                    window_annotations.append({
                        'position': window_center,
                        'text': annot_text,
                        'start_pos': window_positions[0],
                        'end_pos': window_positions[-1]
                    })
            except Exception as e:
                print(f"Warning: Failed to estimate ZINB for window {i+1}: {e}")
                continue
        
        # Add annotations to the plot
        # Stagger vertically to avoid overlap
        y_max = ax.get_ylim()[1]
        annotation_heights = [0.85 * y_max, 0.70 * y_max, 0.55 * y_max]
        
        for idx, annot in enumerate(window_annotations):
            # Cycle through annotation heights
            y_pos = annotation_heights[idx % len(annotation_heights)]
            
            ax.text(
                annot['position'], 
                y_pos,
                annot['text'],
                fontsize=6,
                ha='center',
                va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray', linewidth=0.5)
            )
            
            # Draw a subtle vertical line at window boundaries
            if idx < len(window_annotations) - 1:  # Don't draw after last window
                ax.axvline(annot['end_pos'], color='gray', alpha=0.2, linewidth=0.5, linestyle=':')

    ax.tick_params(axis="x", labelrotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    fig.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    in_dir = "Data/combined_strains/strain_yEK23"
    out_dir = "Data_exploration/plot_SATAY"

    for chrom in CHROMS:
        infile = os.path.join(in_dir, f"Chr{chrom}_distances.csv")
        outfile = os.path.join(out_dir, f"Chr{chrom}_centromere_window.png")

        if not os.path.exists(infile):
            print(f"Skipping (file not found): {infile}")
            continue

        plot_around_centromere_x_is_centdist(
            filepath=infile,
            outpath=outfile,
            n=1000,
            major_tick_step=100,
            minor_tick_step=50,
            strict_centromere_match=True,
            value_max=500,
            window_size=100,  # ZINB parameter estimation window size
            show_zinb_params=True,
        )

        print(f"Saved: {outfile}")