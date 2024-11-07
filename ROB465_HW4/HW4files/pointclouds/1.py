#!/usr/bin/env python
import utils
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### YOUR IMPORTS HERE ###

### YOUR IMPORTS HERE ###


def add_some_outliers(pc, num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc



def icp(source, target, max_iterations=100, tolerance=1e-5):
    """
    Perform Iterative Closest Point algorithm to align source to target.

    Args:
        source (np.ndarray): Source point cloud of shape (N, 3).
        target (np.ndarray): Target point cloud of shape (M, 3).
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance based on change in error.

    Returns:
        transformed_source (np.ndarray): Aligned source point cloud.
        R (np.ndarray): Rotation matrix.
        t (np.ndarray): Translation vector.
        errors (list): List of errors at each iteration.
    """
    # Initialize transformation
    R = np.eye(3)
    t = np.zeros((3, 1))
    transformed_source = source.copy()
    errors = []

    for i in range(max_iterations):
        # Find closest points using Euclidean distance
        distances = np.linalg.norm(target[:, np.newaxis, :] - transformed_source[np.newaxis, :, :], axis=2)
        closest_indices = np.argmin(distances, axis=0)
        closest_points = target[closest_indices]

        # Compute centroids of source and target
        centroid_source = np.mean(transformed_source, axis=0)
        centroid_target = np.mean(closest_points, axis=0)

        # Center the points
        source_centered = transformed_source - centroid_source  # Shape: (N, 3)
        target_centered = closest_points - centroid_target      # Shape: (N, 3)

        # Debug: Print shapes
        print(f"  Iteration {i+1}:")
        print(f"    Source Centered Shape: {source_centered.shape}")
        print(f"    Target Centered Shape: {target_centered.shape}")

        # Compute covariance matrix
        H = source_centered.T @ target_centered  # Shape: (3, 3)

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        R_iter = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R_iter) < 0:
            Vt[2, :] *= -1
            R_iter = Vt.T @ U.T

        # Compute translation
        t_iter = centroid_target.reshape(3, 1) - R_iter @ centroid_source.reshape(3, 1)

        # Update transformation
        R = R_iter @ R
        t = R_iter @ t + t_iter

        # Apply transformation to source
        transformed_source = (R @ source.T).T + t.T  # Shape: (N, 3)

        # Compute mean error
        mean_error = np.mean(np.linalg.norm(transformed_source - closest_points, axis=1))
        errors.append(mean_error)

        print(f"    Mean Error: {mean_error}")

        # Check for convergence
        if i > 0 and abs(errors[-2] - errors[-1]) < tolerance:
            print("error", errors[-2] - errors[-1])
            print("    Convergence reached.")
            break

    return transformed_source, R, t, errors



def main():
    # Import the source and target point clouds
    pc_source = utils.load_pc('cloud_icp_source.csv')
    pc_target = utils.load_pc('cloud_icp_target2.csv')  # Change this to load a different target

    print("Source Point Cloud:")
    print(pc_source)
    print("Target Point Cloud:")
    print(pc_target)

    # Convert point clouds to NumPy arrays for ICP
    # Use .A1 to convert each matrix to a 1-D array of shape (3,)
    source_array = np.array([point.A1 for point in pc_source])  # Shape: (N, 3)
    target_array = np.array([point.A1 for point in pc_target])  # Shape: (M, 3)

    # Verify the shapes
    print(f"Source Array Shape: {source_array.shape}")  # Expected: (N, 3)
    print(f"Target Array Shape: {target_array.shape}")  # Expected: (M, 3)

    # Visualize initial alignment using the original lists of matrices
    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])
    plt.title('Initial Alignment')
    plt.show()

    ### YOUR CODE HERE ###
    # Perform ICP
    transformed_source, R, t, errors = icp(source_array, target_array, max_iterations=10000, tolerance=1e-10)

    # Plot Error vs. Iteration
    plt.figure()
    plt.plot(range(1, len(errors)+1), errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Error')
    plt.title('ICP Error vs. Iteration')
    plt.grid(True)
    plt.savefig('icp_error_vs_iteration.png')  # Save the error plot
    plt.show()

    # Convert transformed_source back to list of (3,1) matrices for visualization
    transformed_source_list = [np.reshape(pt, (3, 1)) for pt in transformed_source]

    # Visualize final alignment using the transformed source
    utils.view_pc([pc_target, transformed_source_list], None, ['r', 'g'], ['^', 'o'])
    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])
    plt.title('Final Alignment after ICP')
    plt.legend(['Target', 'Transformed Source'])
    plt.savefig('icp_final_alignment.png')  # Save the final alignment plot
    plt.show()
    ### YOUR CODE HERE ###

    input("Press enter to end:")


if __name__ == '__main__':
    main()
