import matplotlib.pyplot as plt
from adjustText import adjust_text

# Iterations
iterations = [1, 2, 4, 8]

# Performance data (same as before)
data = {
    'Afterburner-3B-SFT': {
        'Pass@1': [34.33, 18.67, 15.33, 12.00],
        'Time':   [17.19, 12.12, 10.67,  8.03],
        'Memory': [18.42, 12.75, 11.33,  8.51],
        'Integral': [14.31, 10.10,  8.89,  6.69]
    },
    'Afterburner-7B-SFT': {
        'Pass@1': [63.67, 61.00, 53.67, 47.33],
        'Time':   [27.36, 27.13, 26.31, 25.52],
        'Memory': [30.10, 29.84, 28.94, 27.07],
        'Integral': [22.38, 22.61, 22.10, 21.41]
    },
    'Afterburner-3B-DPO': {
        'Pass@1': [42.67, 45.00, 48.00, 48.33],
        'Time':   [18.56, 19.12, 20.04, 20.11],
        'Memory': [28.42, 29.29, 30.66, 31.78],
        'Integral': [15.46, 15.94, 16.70, 17.02]
    },
    'Afterburner-7B-DPO': {
        'Pass@1': [65.67, 69.00, 74.67, 74.67],
        'Time':   [31.51, 33.27, 32.84, 32.19],
        'Memory': [36.24, 38.26, 39.77, 41.02],
        'Integral': [26.25, 27.70, 27.37, 28.51]
    },
    'Afterburner-3B-GRPO': {
        'Pass@1': [50.67, 52.33, 57.33, 60.33],
        'Time':   [35.41, 38.33, 45.59, 49.54],
        'Memory': [39.68, 42.50, 46.21, 50.36],
        'Integral': [27.19, 28.94, 34.55, 39.79]
    }
}

baselines = {
    "3B": {
        "Pass@1": 27.99,
        "Time": 12.40,
        "Memory": 13.24,
        "Integral": 10.29,
    },
    "7B": {
        "Pass@1": 60.78,
        "Time": 27.67,
        "Memory": 29.79,
        "Integral": 21.02
    }
}

palette = ['#EA4335', '#EA4335', '#4285F4', '#4285F4', '#34A853']
size_style = {'3B':'--','7B':'-'}
metrics = ['Pass@1','Time','Memory','Integral']

fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=800)
axs = axs.flatten()

for ax, metric in zip(axs, metrics):
    texts = []
    for i, (label,vals) in enumerate(data.items()):
        size = '3B' if '3B' in label else '7B'
        color = palette[i]
        ax.plot(iterations, vals[metric],
                linestyle=size_style[size],
                color=color,
                label=label,
                marker='o')
    for i, size in enumerate(baselines.keys()):
        ax.axhline(
            y=baselines[size][metric],
            linestyle='--' if size == '3B' else '-',
            color='#FBBC04',
            linewidth=1.0,
            label=f'{size} Baseline'
        )

    ax.set_title(f"{metric} over Iterations")
    ax.set_xticks(iterations)
    ax.grid(True, axis='x')
    
    

# unified legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=7, handlelength=3)

plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig("afterburner_metrics.pdf", dpi=800)
plt.show()