import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Generate data for each distribution
binomial_data = np.random.binomial(n=10, p=0.5, size=1000)
exponential_data = np.random.exponential(scale=2, size=1000)
poisson_data = np.random.poisson(lam=4, size=1000)
uniform_data = np.random.uniform(low=0, high=1, size=1000)
log_data = np.random.lognormal(mean=0, sigma=1, size=1000)
gamma_data = np.random.gamma(shape=2, scale=1, size=1000)
beta_data = np.random.beta(a=2, b=5, size=1000)
chi_square_data = np.random.chisquare(df=2, size=1000)
geometric_data = np.random.geometric(p=0.3, size=1000)

# Define subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# Function to create subplots with transformation
def plot_transformation(ax, data, transformation, title, formula):
    transformed_data = transformation(data)
    sns.histplot(data, kde=True, ax=ax, label='Original Data', alpha=0.5)
    sns.histplot(transformed_data, kde=True, color='red', ax=ax, label='Transformed Data', alpha=0.5)
    ax.set_title(f"{title}\n{formula}")
    ax.legend()

# Binomial - Log Transformation
plot_transformation(
    axes[0, 0],
    binomial_data,
    lambda x: np.log(x + 1),
    'Binomial to Normal',
    r'$log(x + 1)$'
)

# Exponential - Log Transformation
plot_transformation(
    axes[0, 1],
    exponential_data,
    np.log,
    'Exponential to Normal',
    r'$log(x)$'
)

# Poisson - Square Root Transformation
plot_transformation(
    axes[0, 2],
    poisson_data,
    np.sqrt,
    'Poisson to Normal',
    r'$\sqrt{x}$'
)

# Uniform - Rank Transformation
rank_data = stats.rankdata(uniform_data) / len(uniform_data)
normal_data = stats.norm.ppf(rank_data)
sns.histplot(uniform_data, kde=True, ax=axes[1, 0], label='Original Data', alpha=0.5)
sns.histplot(normal_data, kde=True, color='red', ax=axes[1, 0], label='Transformed Data', alpha=0.5)
axes[1, 0].set_title('Uniform to Normal\nRank Transformation')
axes[1, 0].legend()

# Lognormal - Log Transformation
plot_transformation(
    axes[1, 1],
    log_data,
    np.log,
    'Lognormal to Normal',
    r'$log(x)$'
)

# Gamma - Box-Cox Transformation
boxcox_data, _ = stats.boxcox(gamma_data)
sns.histplot(gamma_data, kde=True, ax=axes[1, 2], label='Original Data', alpha=0.5)
sns.histplot(boxcox_data, kde=True, color='red', ax=axes[1, 2], label='Transformed Data', alpha=0.5)
axes[1, 2].set_title('Gamma to Normal\nBox-Cox Transformation')
axes[1, 2].legend()

# Beta - Arcsine Transformation
plot_transformation(
    axes[2, 0],
    beta_data,
    lambda x: np.arcsin(np.sqrt(x)),
    'Beta to Normal',
    r'$arcsin(\sqrt{x})$'
)

# Chi-Square - Square Root Transformation
plot_transformation(
    axes[2, 1],
    chi_square_data,
    np.sqrt,
    'Chi-Square to Normal',
    r'$\sqrt{x}$'
)

# Geometric - Log Transformation
plot_transformation(
    axes[2, 2],
    geometric_data,
    np.log,
    'Geometric to Normal',
    r'$log(x)$'
)

plt.show()
