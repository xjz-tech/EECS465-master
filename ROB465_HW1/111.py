import numpy as np

extrinsic_matrix = np.array([[9.99816753e-01, -1.11207014e-02, -1.55817591e-02, 2.80835024e+01],
                             [-8.57851459e-03, -9.87933558e-01, 1.54640532e-01, 1.05196846e+02],
                             [-1.71134539e-02, -1.54478526e-01, -9.87847921e-01, 1.02988628e+03 + 5],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

intrinsic_matrix = np.array([[910.6915, 0, 642.47],
                             [0, 909.0733, 398.6077],
                             [0, 0, 1]])

world_point = np.array([-450, -125, 0, 1])

# Convert world coordinate to camera coordinate using the extrinsic matrix
camera_coords = np.dot(extrinsic_matrix, world_point)

# Convert camera coordinate to pixel coordinate using the intrinsic matrix
camera_coords_normalized = camera_coords[:3] / camera_coords[2]  # Normalize by the Z value
pixel_coords_homogeneous = np.dot(intrinsic_matrix, camera_coords_normalized)
pixel_coords = pixel_coords_homogeneous[:2] / pixel_coords_homogeneous[2]

print("Pixel Coordinates:", pixel_coords)

# Convert pixel coordinates back to camera coordinates
pixel_coords_homogeneous_back = np.array([pixel_coords[0], pixel_coords[1], 1])
intrinsic_inv = np.linalg.inv(intrinsic_matrix)
camera_coords_back = np.dot(intrinsic_inv, pixel_coords_homogeneous_back)

#Convert camera coordinates back to world coordinates
camera_coords_back_homogeneous = np.array([camera_coords_back[0] * camera_coords[2],
                                           camera_coords_back[1] * camera_coords[2],
                                           camera_coords[2], 1])

#
extrinsic_inv = np.linalg.inv(extrinsic_matrix)
world_coords_back = np.dot(extrinsic_inv, camera_coords_back_homogeneous)

print("World Coordinates Back:", world_coords_back[:3])
