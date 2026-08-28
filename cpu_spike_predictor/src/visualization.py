import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

# Set style parameters for academic/industry report quality
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

def create_mtta_visualizations(mtta_results_path: str = "outputs/cpu_mtta_results.csv", output_path: str = "outputs/cpu_mtta_chart.png"):
    """
    Generates high-quality visualizations of the MTTA results and strategy comparison.
    Creates a dual-panel figure:
      1. Strategy Comparison Bar Chart (Mean MTTA & Anticipation Rate)
      2. Project-level MTTA Box Plot for the CPU_Only strategy
    """
    if not os.path.exists(mtta_results_path):
        raise FileNotFoundError(f"MTTA results file not found at: {mtta_results_path}")
        
    df = pd.read_csv(mtta_results_path)
    
    # Create the figure with dual subplots (side-by-side or stacked)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"hspace": 0.4})
    
    # Palette definition (modern curated colors: Teal, Slate Blue, Coral, Purple)
    strategy_colors = {
        "CPU_Only": "#0984e3",     # Vibrant Blue
        "Memory_Only": "#e17055",   # Coral
        "Joint_AND": "#6c5ce7",     # Soft Purple
        "Joint_OR": "#00b894"       # Mint Teal
    }
    
    # -------------------------------------------------------------------------
    # Panel 1: Strategy Comparison (Mean MTTA for anticipated events)
    # -------------------------------------------------------------------------
    # Filter for anticipated events to compute mean warning time
    anticipated_df = df[df["anticipated"] == 1]
    
    strategy_summary = anticipated_df.groupby("strategy")["mtta_minutes"].agg(["mean", "std", "count"]).reset_index()
    
    # Join with overall counts to calculate anticipation rate
    total_counts = df.groupby("strategy")["anticipated"].count().reset_index().rename(columns={"anticipated": "total"})
    strategy_summary = pd.merge(strategy_summary, total_counts, on="strategy")
    strategy_summary["anticipation_rate"] = (strategy_summary["count"] / strategy_summary["total"]) * 100.0
    
    # Sort for consistent display
    strategy_summary = strategy_summary.set_index("strategy").reindex(["CPU_Only", "Memory_Only", "Joint_AND", "Joint_OR"]).reset_index()
    
    ax1 = axes[0]
    bars = ax1.bar(
        strategy_summary["strategy"], 
        strategy_summary["mean"], 
        yerr=strategy_summary["std"].fillna(0),
        color=[strategy_colors[s] for s in strategy_summary["strategy"]],
        edgecolor="#2d3436",
        linewidth=0.8,
        width=0.5,
        capsize=5,
        error_kw={"elinewidth": 1.5, "ecolor": "#2d3436"}
    )
    
    ax1.set_ylabel("Mean Warning Time (Minutes)", fontsize=11, fontweight="bold", labelpad=10)
    ax1.set_title("Alarm Anticipation Strategy Comparison (Mean MTTA)", fontsize=13, fontweight="bold", pad=15)
    ax1.set_ylim(0, max(strategy_summary["mean"]) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, color="#CCCCCC")
    ax1.set_axisbelow(True)
    
    # Remove top and right spines
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
        
    # Annotate bars with values and anticipation rates
    for bar, (_, row) in zip(bars, strategy_summary.iterrows()):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.0, 
            height + (max(strategy_summary["mean"]) * 0.03), 
            f"{height:.2f} mins\n({row['anticipation_rate']:.0f}% detect)", 
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2d3436"
        )
        
    # -------------------------------------------------------------------------
    # Panel 2: Distribution of MTTA per project/service (CPU_Only strategy)
    # -------------------------------------------------------------------------
    cpu_mtta = anticipated_df[anticipated_df["strategy"] == "CPU_Only"].copy()
    # Sort projects numerically Proj_01 -> Proj_10
    cpu_mtta = cpu_mtta.sort_values(by="project_id")
    
    ax2 = axes[1]
    
    # Boxplot with swarmplot overlay for micro-distribution
    sns.boxplot(
        x="project_id", 
        y="mtta_minutes", 
        data=cpu_mtta, 
        ax=ax2, 
        color="#74b9ff", 
        width=0.4,
        boxprops=dict(edgecolor="#2d3436", linewidth=1.0),
        whiskerprops=dict(color="#2d3436", linewidth=1.0),
        capprops=dict(color="#2d3436", linewidth=1.0),
        medianprops=dict(color="#d63031", linewidth=1.5)  # Red median line
    )
    
    sns.swarmplot(
        x="project_id", 
        y="mtta_minutes", 
        data=cpu_mtta, 
        ax=ax2, 
        color="#2d3436", 
        size=6, 
        alpha=0.8
    )
    
    ax2.set_xlabel("Microservice Project ID", fontsize=11, fontweight="bold", labelpad=10)
    ax2.set_ylabel("MTTA (Minutes)", fontsize=11, fontweight="bold", labelpad=10)
    ax2.set_title("Distribution of Anticipation Warning Time (MTTA) per Project (CPU_Only)", fontsize=13, fontweight="bold", pad=15)
    ax2.set_ylim(0, 6.0) # bounded by 5-min lead time
    ax2.grid(axis="y", linestyle="--", alpha=0.5, color="#CCCCCC")
    ax2.set_axisbelow(True)
    
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
        
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Generated premium visualization chart at: {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        create_mtta_visualizations()
    except Exception as e:
        logger.error(f"Visualization script failed: {e}")
