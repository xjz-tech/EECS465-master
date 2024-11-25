#!/usr/bin/env python
import utils
import numpy as np
import time
import random
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
def compute_pca_plane(pc):
    # Start timing
    start_time = time.time()

    # Convert to NumPy array and ensure shape is (n, 3)
    pc_array = np.array([point.flatten() for point in pc])
    if pc_array.ndim == 3:
        pc_array = pc_array.squeeze(axis=1)  # Remove the singleton dimension
    elif pc_array.ndim != 2 or pc_array.shape[1] != 3:
        raise ValueError(f"Unexpected pc_array shape: {pc_array.shape}")

    # Calculate mean
    pc_mean = np.mean(pc_array, axis=0)
    # Center the data
    pc_centered = pc_array - pc_mean

    # Calculate covariance matrix
    cov_matrix = np.cov(pc_centered, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Normal vector corresponds to the eigenvector with the smallest eigenvalue
    normal = eigenvectors[:, 0]
    # Plane equation parameter
    d = -pc_mean.dot(normal)
    # End timing
    end_time = time.time()
    computation_time = end_time - start_time
    return normal, d, computation_time


def compute_ransac_plane(pc):
    # Start timing
    start_time = time.time()
    data = np.array([point.flatten() for point in pc])
    if data.ndim == 3:
        data = data.squeeze(axis=1)  # Remove the singleton dimension
    elif data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Unexpected data shape in RANSAC: {data.shape}")

    max_iterations = 1000
    distance_threshold = 0.05
    num_points = data.shape[0]
    best_inliers = []
    best_plane = None

    for _ in range(max_iterations):
        # Randomly select 3 points
        indices = random.sample(range(num_points), 3)
        sample_points = data[indices]
        p1, p2, p3 = sample_points

        # Calculate normal vector
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)

        # Calculate distances
        distances = np.abs(np.dot(data, normal) + d)
        inlier_indices = np.where(distances < distance_threshold)[0]
        if len(inlier_indices) > len(best_inliers):
            best_inliers = inlier_indices
            best_plane = (normal, d)

    # Refine the plane using inliers
    if best_plane is not None and len(best_inliers) >= 3:
        inlier_points = data[best_inliers]
        pc_mean = np.mean(inlier_points, axis=0)
        pc_centered = inlier_points - pc_mean
        cov_matrix = np.cov(pc_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, 0]
        d = -pc_mean.dot(normal)
    else:
        normal = None
        d = None
    # End timing
    end_time = time.time()
    computation_time = end_time - start_time
    return normal, d, best_inliers, computation_time


def compute_error(pc, normal, d, inlier_indices):
    data = np.array([point.flatten() for point in pc])
    if data.ndim == 3:
        data = data.squeeze(axis=1)  # Remove the singleton dimension
    distances = np.abs(np.dot(data, normal) + d)
    error = np.sum(distances[inlier_indices] ** 2)
    return error

num_outliers_list = []
pca_errors = []
ransac_errors = []
pca_times = []
ransac_times = []
###YOUR IMPORTS HERE###

def add_some_outliers(pc,num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc


def main():
    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    num_tests = 10
    fig = None
    for i in range(0,num_tests):
        pc = add_some_outliers(pc,10) #adding 10 new outliers for each test
        # fig = utils.view_pc([pc])

        ###YOUR CODE HERE###

        num_outliers = (i + 1) * 10
        num_outliers_list.append(num_outliers)

        # PCA algorithm
        try:
            pca_normal, pca_d, pca_time = compute_pca_plane(pc)
        except ValueError as e:
            print(f"PCA Error at iteration {i + 1}: {e}")
            continue
        data = np.array([point.flatten() for point in pc])
        if data.ndim == 3:
            data = data.squeeze(axis=1)
        distances_pca = np.abs(np.dot(data, pca_normal) + pca_d)
        # Determine inliers
        distance_threshold = 0.05
        pca_inlier_indices = np.where(distances_pca < distance_threshold)[0]
        # print(distances_pca[np.where(distances_pca < distance_threshold)[0]])
        pca_error = np.sum(distances_pca[pca_inlier_indices] ** 2)
        pca_errors.append(pca_error)
        pca_times.append(pca_time)

        # RANSAC algorithm
        try:
            ransac_normal, ransac_d, ransac_inliers, ransac_time = compute_ransac_plane(pc)
            if ransac_normal is None:
                raise ValueError("RANSAC failed to find a valid plane.")
        except ValueError as e:
            print(f"RANSAC Error at iteration {i + 1}: {e}")
            continue
        distances_ransac = np.abs(np.dot(data, ransac_normal) + ransac_d)
        ransac_error = np.sum(distances_ransac[ransac_inliers] ** 2)
        ransac_errors.append(ransac_error)
        ransac_times.append(ransac_time)

        # Generate plots on the last iteration
        if i == num_tests - 1:
            # **PCA result plot**
            fig_pca = plt.figure()
            ax_pca = fig_pca.add_subplot(111, projection='3d')

            # **Plot PCA inliers (red)**
            pca_inliers = data[pca_inlier_indices]
            ax_pca.scatter(pca_inliers[:, 0], pca_inliers[:, 1], pca_inliers[:, 2],
                           c='red', label='Inliers')

            # **Plot PCA outliers (blue)**
            pca_outlier_indices = np.array([idx for idx in range(len(data)) if idx not in pca_inlier_indices])
            pca_outliers = data[pca_outlier_indices]
            ax_pca.scatter(pca_outliers[:, 0], pca_outliers[:, 1], pca_outliers[:, 2],
                           c='blue', label='Outliers')

            # **Plot PCA plane (green)**
            xx, yy = np.meshgrid(np.linspace(data[:, 0].min(), data[:, 0].max(), 10),
                                 np.linspace(data[:, 1].min(), data[:, 1].max(), 10))
            if pca_normal[2] != 0:
                zz = (-pca_normal[0] * xx - pca_normal[1] * yy - pca_d) / pca_normal[2]
                ax_pca.plot_surface(xx, yy, zz, alpha=0.5, color='green')
            else:
                print("PCA Plane is vertical. Cannot plot z values.")

            ax_pca.set_title('PCA Plane Fitting')
            ax_pca.legend()
            plt.savefig('pca_plane_fitting.png')  # Save the figure
            plt.show()

            # **RANSAC result plot**
            fig_ransac = plt.figure()
            ax_ransac = fig_ransac.add_subplot(111, projection='3d')

            # **Plot RANSAC inliers (red)**
            ransac_inliers_points = data[ransac_inliers]
            ax_ransac.scatter(ransac_inliers_points[:, 0], ransac_inliers_points[:, 1],
                              ransac_inliers_points[:, 2], c='red', label='Inliers')

            # **Plot RANSAC outliers (blue)**
            ransac_outlier_indices = np.array([idx for idx in range(len(data)) if idx not in ransac_inliers])
            ransac_outliers = data[ransac_outlier_indices]
            ax_ransac.scatter(ransac_outliers[:, 0], ransac_outliers[:, 1], ransac_outliers[:, 2],
                              c='blue', label='Outliers')

            # **Plot RANSAC plane (green)**
            if ransac_normal[2] != 0:
                zz = (-ransac_normal[0] * xx - ransac_normal[1] * yy - ransac_d) / ransac_normal[2]
                ax_ransac.plot_surface(xx, yy, zz, alpha=0.5, color='green')
            else:
                print("RANSAC Plane is vertical. Cannot plot z values.")

            ax_ransac.set_title('RANSAC Plane Fitting')
            ax_ransac.legend()
            plt.savefig('ransac_plane_fitting.png')  # Save the figure
            plt.show()

    # **Requirement (b): Plot error vs number of outliers**
    plt.figure()
    plt.plot(num_outliers_list, pca_errors, label='PCA Error', marker='o')
    plt.plot(num_outliers_list, ransac_errors, label='RANSAC Error', marker='x')
    plt.xlabel('Number of Outliers')
    plt.ylabel('Error')
    plt.title('Error vs Number of Outliers')
    plt.legend()
    plt.grid(True)
    plt.savefig('error_vs_outliers.png')  # Save as image file
    plt.show()

    # **Plot computation time vs number of outliers**
    plt.figure()
    plt.plot(num_outliers_list, pca_times, label='PCA Time', marker='o')
    plt.plot(num_outliers_list, ransac_times, label='RANSAC Time', marker='x')
    plt.xlabel('Number of Outliers')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Computation Time vs Number of Outliers')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_vs_outliers.png')  # Save as image file
    plt.show()
        ###YOUR CODE HERE###

    input("Press enter to end")


if __name__ == '__main__':
    main()
