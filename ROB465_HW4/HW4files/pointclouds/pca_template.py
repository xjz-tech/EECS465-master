#!/usr/bin/env python
import utils
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools explicitly

###YOUR IMPORTS HERE###


def main():

    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    ###YOUR CODE HERE###
    # fig = utils.view_pc([pc])
    pc = np.array(pc)

    # Center the data
    pc_mean = pc.mean(axis=0, keepdims=True)  # Keep shape as (1, 3, 1)
    pc_centered = pc - pc_mean  # Subtract mean, shape remains (200, 3, 1)

    # Compute covariance matrix (flatten to 2D for PCA computation)
    pc_flattened = pc_centered.squeeze(-1)  # Shape (200, 3)
    cov_matrix = np.cov(pc_flattened, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Rotate the point cloud (apply rotation matrix)
    pc_rotated = (pc_flattened @ eigenvectors).reshape(pc.shape)

    #Show the resulting point cloud
    fig = utils.view_pc([pc_rotated])  # Shape: (200, 3, 1)
    ax = fig.axes[0]
    ax.view_init(elev=90, azim=-90)  # Align with XY plane
    plt.title("Part a: Rotated Point Cloud")
    print("Part a) V^T matrix applied:")
    print(eigenvectors.T)
    plt.show()  # Display the plot and wait until it's closed


    # Part b: Rotate the points and eliminate noise
    n_components = 2  # Keep the first two principal components
    P = eigenvectors[:, :n_components] @ eigenvectors[:, :n_components].T  # Projection matrix

    print("Part b) Projection matrix P (V^T applied):")
    print(P)

    # Project the data (denoising and dimensionality reduction)
    pc_denoised = (pc_flattened @ P).reshape(pc.shape)

    # Add back the mean
    pc_denoised += pc_mean

    pc_denoised[:, 2, :] = 0  # Explicitly set z values to 0



    # Show the resulting point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter(pc_denoised[:, 0, 0], pc_denoised[:, 1, 0], pc_denoised[:, 2, 0], c='blue', marker='o')
    ax.view_init(elev=90, azim=-90)  # Set the view from above to show the point cloud flattened on the XY plane
    plt.title("Part b: Denoised 2D Point Cloud (Projected onto XY Plane)")
    plt.show()


    # Part c: Fit a plane to the cloud and draw it
    fig = utils.view_pc([pc])
    ax = fig.axes[0]

    # Plane fitting
    normal_vector = eigenvectors[:, 2]
    mean_point = pc_mean.squeeze()
    x0, y0, z0 = mean_point

    # Grid for the plane
    xlim = [pc[:, 0, 0].min(), pc[:, 0, 0].max()]
    ylim = [pc[:, 1, 0].min(), pc[:, 1, 0].max()]
    x_range = np.linspace(xlim[0], xlim[1], num=10)
    y_range = np.linspace(ylim[0], ylim[1], num=10)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Calculate z values for the plane
    a, b, c = normal_vector
    if abs(c) > 1e-6:
        z_grid = (-a * (x_grid - x0) - b * (y_grid - y0)) / c + z0
    else:
        z_grid = np.full_like(x_grid, z0)

    # Plot the plane
    ax.plot_surface(x_grid, y_grid, z_grid, color='green', alpha=0.5)
    plt.title("Part c: Fitted Plane with Point Cloud")
    ###YOUR CODE HERE###








    plt.show()
    #input("Press enter to end:")



if __name__ == '__main__':
    main()
