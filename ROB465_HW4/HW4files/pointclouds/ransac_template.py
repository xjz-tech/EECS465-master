#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import random
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')


    ###YOUR CODE HERE###
    # Convert point cloud to a NumPy array and flatten the points
    # Ensure that each point is a 1D array with 3 elements
    data = np.array([point.flatten() for point in pc])  # Initial shape: (num_points, 3)
    print("Initial data shape:", data.shape)  # Debugging statement

    # Check and remove any singleton dimensions
    if data.ndim == 3 and data.shape[1] == 1 and data.shape[2] == 3:
        data = data.reshape(data.shape[0], data.shape[2])
        print("Data reshaped to:", data.shape)  # Debugging statement
    elif data.ndim == 2 and data.shape[1] == 3:
        print("Data is already in the correct shape.")
    else:
        print("Unexpected data shape:", data.shape)
        return  # Exit if data is not in the expected shape

    # Show the input point cloud
    utils.view_pc([pc])

    # RANSAC parameters
    max_iterations = 1000
    distance_threshold = 0.01
    num_points = data.shape[0]
    best_inliers = []
    best_plane = None

    for i in range(max_iterations):
        # Randomly select 3 unique indices
        indices = random.sample(range(num_points), 3)
        sample_points = data[indices]

        # Extract the three points and ensure they are 1D arrays
        p1, p2, p3 = sample_points
        p1 = p1.flatten()
        p2 = p2.flatten()
        p3 = p3.flatten()
        # Uncomment the following line for debugging
        # print(f"Sample points:\np1: {p1}\np2: {p2}\np3: {p3}")

        # Compute two vectors in the plane
        v1 = p2 - p1  # Shape: (3,)
        v2 = p3 - p1  # Shape: (3,)

        # Compute the normal vector of the plane
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            continue  # Skip degenerate cases where points are colinear
        normal = normal / normal_norm  # Normalize the normal vector
        # Ensure it's a 1D array
        normal = normal.flatten()
        # Uncomment the following line for debugging
        # print(f"Normal vector: {normal}")

        # Plane equation: ax + by + cz + d = 0
        a, b, c = normal  # Unpack the normal vector
        d = -np.dot(normal, p1)  # Compute d using p1
        # Uncomment the following line for debugging
        # print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        # Compute distances from all points to the plane
        distances = np.abs(np.dot(data, normal) + d)

        # Determine inliers based on the distance threshold
        inlier_indices = np.where(distances < distance_threshold)[0]
        inliers = data[inlier_indices]

        # Keep the plane with the largest number of inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (a, b, c, d)
            best_inlier_indices = inlier_indices

    # Refine the plane using all inliers
    if best_plane is not None and len(best_inliers) >= 3:
        # Subtract the mean of inliers
        mean_inliers = np.mean(best_inliers, axis=0)
        centered_inliers = best_inliers - mean_inliers
        print("Centered inliers shape:", centered_inliers.shape)  # Debugging statement

        # Compute the covariance matrix
        try:
            cov_matrix = np.cov(centered_inliers, rowvar=False)
            print("Covariance matrix shape:", cov_matrix.shape)  # Debugging statement
        except ValueError as e:
            print("Error computing covariance matrix:", e)
            return

        # Perform eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            print("Eigenvalues:", eigenvalues)  # Debugging statement
            print("Eigenvectors shape:", eigenvectors.shape)  # Debugging statement
        except np.linalg.LinAlgError as e:
            print("Error performing eigenvalue decomposition:", e)
            return

        # The normal vector is the eigenvector corresponding to the smallest eigenvalue
        normal = eigenvectors[:, 0]  # Eigenvalues are sorted in ascending order
        print("Normal vector before flattening:", normal)  # Debugging statement

        # Ensure the normal vector is a 1D array
        normal = normal.flatten()
        print("Normal vector after flattening:", normal)  # Debugging statement

        # Ensure the normal vector is a unit vector
        normal = normal / np.linalg.norm(normal)
        print("Normal vector after normalization:", normal)  # Debugging statement

        # Plane passes through the mean of inliers
        a, b, c = normal
        d = -np.dot(normal, mean_inliers)

        # Update the best plane parameters
        best_plane = (a, b, c, d)
    else:
        print("Not enough inliers to refine the plane.")
        return  # Exit the function if we cannot refine the plane

    # Output the plane equation
    print(f"The equation of the plane is: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # Show the resulting point cloud with inliers and outliers
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot inliers in red and outliers in blue
    outlier_indices = np.array([i for i in range(num_points) if i not in best_inlier_indices])
    outliers = data[outlier_indices]
    print("Outliers shape:", outliers.shape)  # Debugging statement
    print("Inliers shape:", best_inliers.shape)  # Debugging statement

    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c='b', marker='o', label='Outliers')
    ax.scatter(best_inliers[:, 0], best_inliers[:, 1], best_inliers[:, 2], c='r', marker='o', label='Inliers')

    # Create a grid to plot the plane
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 10),
        np.linspace(ylim[0], ylim[1], 10)
    )

    # Compute the corresponding z values for the plane
    a, b, c, d = best_plane
    if c != 0:
        zz = (-a * xx - b * yy - d) / c
        ax.plot_surface(xx, yy, zz, alpha=0.5, color='green')
    else:
        # Handle the case where the plane is vertical (c == 0)
        print("The plane is vertical. Cannot plot z values.")
        return

    plt.title('Plane Fitting Using RANSAC')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    #Fit a plane to the data using ransac



    #Show the resulting point cloud

    #Draw the fitted plane


    ###YOUR CODE HERE###
    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
