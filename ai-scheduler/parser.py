import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset (make sure the file is in same directory)
df = pd.read_csv('5g-channelAllocationSummary.csv')

# Clean column names (handles accidental spaces in CSV)
df.columns = df.columns.str.strip()

# Verify structure
print(df.head())
print(df.info())

# Extract required columns
services = df['Service']
rr_channels = df['RR_Channels']
pf_channels = df['PF_Channels']
ml_channels = df['ML_Channels']

# X-axis positions
x = np.arange(len(services))
width = 0.25

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width, rr_channels, width, label='RR Channels')
rects2 = ax.bar(x, pf_channels, width, label='PF Channels')
rects3 = ax.bar(x + width, ml_channels, width, label='ML Channels')

# Labels and title
ax.set_xlabel('Service Type')
ax.set_ylabel('Channels Allocated')
ax.set_title('Channel Allocation by Scheduler and Service Type')
ax.set_xticks(x)
ax.set_xticklabels(services)

# Add value labels on bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

ax.legend()

plt.tight_layout()
plt.savefig('5G-channelAllocation.png')
plt.show()