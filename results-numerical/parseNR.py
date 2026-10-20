import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

files = glob.glob("FMResultsNR/FMResults_*.csv")

data_summary = []

for file in files:
    match = re.search(r"FMResults_(\w+)_(\d+)\.csv", file)
    if not match:
        continue

    scheduler = match.group(1)
    n_ues = int(match.group(2))

    df = pd.read_csv(file)

    # Normalize column names for compatibility (NR vs LTE)
    df.columns = df.columns.str.strip()

    column_mapping = {
        "AvgDelay(ms)": "Delay(ms)",
        "AvgJitter(ms)": "Jitter(ms)"
    }

    df = df.rename(columns=column_mapping)

    df = df[df["UE_ID"] != -1]

    avg_throughput = df["Throughput(Mbps)"].mean()
    system_throughput = df["Throughput(Mbps)"].sum()
    avg_delay = df["Delay(ms)"].mean()
    avg_jitter = df["Jitter(ms)"].mean()
    packet_loss = df["LostPackets"].sum() / (df["TxPackets"].sum() + 1e-9)
    ue_throughput = df.groupby("UE_ID")["Throughput(Mbps)"].mean()
    fairness = (ue_throughput.sum() ** 2) / (len(ue_throughput) * (ue_throughput ** 2).sum() + 1e-9)

    data_summary.append({
        "Scheduler": scheduler,
        "UEs": n_ues,
        "Throughput": avg_throughput,
        "SystemThroughput": system_throughput,
        "AvgUEThroughput": ue_throughput,
        "Delay": avg_delay,
        "Jitter": avg_jitter,
        "PacketLoss": packet_loss,
        "Fairness": fairness,
    })

summary_df = pd.DataFrame(data_summary)

summary_df = summary_df.sort_values(by=["Scheduler", "UEs"])

fig, axs = plt.subplots(2, 3, figsize=(18, 10))

plots = [
    ("SystemThroughput", "System Throughput (Mbps)"),
    ("Throughput", "Average Throughput (Mbps)"),
    ("Delay", "Average Delay (ms)"),
    ("Jitter", "Average Jitter (ms)"),
    ("PacketLoss", "Packet Loss Ratio"),
    ("Fairness", "Jain Fairness Index")
]

lines = []
labels = []

for ax, (metric, ylabel) in zip(axs.flatten(), plots):
    for scheduler in summary_df["Scheduler"].unique():
        subset = summary_df[summary_df["Scheduler"] == scheduler]

        line, = ax.plot(
            subset["UEs"],
            subset[metric],
            marker='o',
            linewidth=2,
            markersize=6,
            label=scheduler
        )

        if scheduler not in labels:
            lines.append(line)
            labels.append(scheduler)

    ax.set_xlabel("Number of UEs", fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_title(ylabel, fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)

fig.legend(
    lines,
    labels,
    loc='lower center',
    ncol=4,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02)
)

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig("FMResults.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nPerformance Summary: \n")

rows = []

for u in sorted(summary_df["UEs"].unique()):
    subset = summary_df[summary_df["UEs"] == u]

    best_throughput = subset.loc[subset["Throughput"].idxmax()]
    best_delay = subset.loc[subset["Delay"].idxmin()]
    best_loss = subset.loc[subset["PacketLoss"].idxmin()]

    rows.append({
        "UEs": u,
        "Best Throughput": f"{best_throughput['Scheduler']} ({best_throughput['Throughput']:.4f})",
        "Lowest Delay": f"{best_delay['Scheduler']} ({best_delay['Delay']:.2f} ms)",
        "Lowest Loss": f"{best_loss['Scheduler']} ({best_loss['PacketLoss']:.6f})"
    })

best_df = pd.DataFrame(rows)

print("\t\t".join(best_df.columns))
for _, row in best_df.iterrows():
    print("\t\t".join(map(str, row)))