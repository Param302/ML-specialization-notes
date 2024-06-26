import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import multivariate_normal

# Parameters to set
mu_x = 5
variance_x = 4

mu_y = 3
variance_y = 1

# Create grid and multivariate normal
x = np.linspace(-2, 12, 800)
y = np.linspace(-2, 12, 800)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])


# Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos), cmap='coolwarm',
                linewidth=0, alpha=0.7, zorder=-1, label='PDF')
ax.set_title('Normal Distribution of $p(x_1, x_2)$')
ax.set_xlabel('$p(x_1)$')
ax.set_ylabel('$p(x_2)$')
ax.set_zlabel('$p(x_1, x_2)$')

# Plot 2 scatter points in the 3D plot
x1 = x.mean()
x2 = y.mean()
ax.scatter(x1, x2, rv.pdf([x1, x2]), color='red', s=100, zorder=10,
           label=f"$x_1 = {int(x.mean())}, x_2 = {int(y.mean())}$")
x1 = 4
x2 = 4
ax.scatter(x1, x2, rv.pdf([x1, x2]), color='green',
           s=100, zorder=10, label=f"$x_1 = {x1}, x_2 = {x2}$")

custom_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=10, label=f"$x_1 = ({int(x.mean())}, {int(y.mean())})$"),
    Line2D([0], [0], linestyle='none', marker='o', color='w',
           markerfacecolor='green', markersize=10, label=f"$x_2 = ({x1}, {x2})$")
]
ax.legend(handles=custom_legend, loc="upper right")

plt.show()
