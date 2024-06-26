import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Generate data for each distribution
exponential_data = np.random.exponential(scale=2, size=1000)
beta_data = np.random.beta(a=2, b=5, size=1000)

# Define subplots with adjusted width ratios
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10),
                         gridspec_kw={'width_ratios': [3, 1, 3]})
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# Function to create subplots with transformation


def plot_transformation(ax, data, transformation, title):
    transformed_data = transformation(data)
    sns.histplot(data, kde=True, ax=ax[0],
                 label='Original Data', alpha=0.5, bins=50)
    sns.histplot(transformed_data, kde=True, color='red',
                 ax=ax[2], label='Transformed Data', alpha=0.5, bins=50)
    ax[0].set_title(f"{title} Distribution")
    ax[2].set_title("Normal Distribution")
    ax[1].set_title("Transformation", fontsize=18)
    ax[1].set_xticks([])
    ax[1].set_yticks([])


# Exponential - Log Transformation
plot_transformation(
    axes[0],
    exponential_data,
    np.log,
    'Exponential'
)

# Empty subplot with formula
axes[0, 1].axis('off')
axes[0, 1].text(0.5, 0.55, r'$log(x)$', horizontalalignment='center',
                verticalalignment='center', fontsize=16)
axes[0, 1].text(0.5, 0.5, r"$\longrightarrow$",
                horizontalalignment='center', verticalalignment='center', fontsize=40)

# Beta - Arcsine Transformation
plot_transformation(
    axes[1],
    beta_data,
    lambda x: np.arcsin(np.sqrt(x)),
    'Beta'
)

# Empty subplot with formula
axes[1, 1].axis('off')
axes[1, 1].text(
    0.5, 0.55, r'$arcsin(\sqrt{x})$', horizontalalignment='center', verticalalignment='center', fontsize=16)
axes[1, 1].text(0.5, 0.5, r"$\longrightarrow$",
                horizontalalignment='center', verticalalignment='center', fontsize=40)

axes[1, 1].set_title("")
axes[1, 2].set_title("")

plt.tight_layout()
plt.show()
