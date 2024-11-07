import argparse

from world import World
from utils import NUM_FRICTION_CONE_VECTORS, process_contact_point, get_f0_pre_rotation

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

def calculate_wrenches(contact_points, world):
    """
    Calculate the 6D wrenches for each contact point.
    :param contact_points: A list of contact points. Each contact point is a tuple of information from p.getContactPoints.
    :param world: The world object.
    :return: A numpy array of shape (NUM_FRICTION_CONE_VECTORS*n, 6) where n is the number of contact points.
    """
    
    #Get the object position and friction coefficient
    position, mu, max_radius= world.get_object_info()
    wrenches = []

    for contact_point in contact_points:
        contact_pos, contact_force_vector, tangent_dir = process_contact_point(contact_point)
        cone_edges = calculate_friction_cone(contact_force_vector, tangent_dir, mu)

        if cone_edges is not None:
            for i in range(NUM_FRICTION_CONE_VECTORS):
                wrench = np.zeros(6)
                cone_vector = cone_edges[i]
                wrench[:3] = cone_vector

                #Calculate the radius vector
                radius = contact_pos - position

                #Calculate the torque
                torque = np.cross(radius, cone_vector)
                #Scale the torque by the max radius so that the units are consistent with the force
                torque /= max_radius
                wrench[3:] = torque
                wrenches.append(wrench)

    wrenches = np.array(wrenches)
    return wrenches

def calculate_friction_cone(contact_force_vector, tangent_dir, mu, num_cone_vectors=NUM_FRICTION_CONE_VECTORS):
    """
    Calculate the friction cone vectors for a single contact point.
    :param contact_force_vector: The contact force vector.
    :param tangent_dir: A vector tangent to the contact surface.
    :param mu: The coefficient of friction.
    :return: A numpy array of shape (NUM_FRICTION_CONE_VECTORS, 3) where each row is a vector in the friction cone.
    """
    
    cone_edges = None
    _, contact_unit_normal, f_0_pre_rotation = get_f0_pre_rotation(contact_force_vector, mu)
    ### YOUR CODE HERE ###

    # Compute the angle beta based on the friction coefficient
    beta = np.arctan(mu)

    # Rotate the contact force vector about the tangent direction by beta to get the initial f_0 vector
    rotation_about_tangent = Rotation.from_rotvec(beta * tangent_dir)
    f_0 = rotation_about_tangent.apply(f_0_pre_rotation)

    cone_edges = []  # Start with an empty list for storing cone vectors
    for i in range(num_cone_vectors):
        # Evenly spaced rotation around the contact normal
        angle = 2 * np.pi * i / num_cone_vectors
        rotation_around_normal = Rotation.from_rotvec(angle * contact_unit_normal)
        cone_vector = rotation_around_normal.apply(f_0)  # Apply the rotation to f_0

        # Debugging prints
        # print("contact_unit_normal", contact_unit_normal)
        # print("angle * contact_unit_normal", angle * contact_unit_normal)
        # print("f_0", f_0)
        # print("cone vector", cone_vector)

        cone_edges.append(cone_vector)  # Add the resulting vector to the list

    cone_edges = np.array(cone_edges)  # Convert the list to a numpy array


    return cone_edges

def compare_discretization(contact_point, world):
    """
    Calculate the volume of the friction cone using different discretizations.
    :param contact_point: A contact point. This is a tuple of information from p.getContactPoints.
    :param world: The world object.
    :return: None
    """
    # print("*" * 50)
    _, mu, _= world.get_object_info()
    _, contact_force_vector, tangent_dir = process_contact_point(contact_point)

    four_vector = calculate_friction_cone(contact_force_vector, tangent_dir, mu, num_cone_vectors=4)
    eight_vector = calculate_friction_cone(contact_force_vector, tangent_dir, mu, num_cone_vectors=8)

    if four_vector is None or eight_vector is None:
        print('calculate_friction_cone not implemented')
    
    true_volume = None
    four_vector_volume = None
    eight_vector_volume = None

    ### YOUR CODE HERE ###
    h = np.linalg.norm(contact_force_vector)  # Assume the height of the cone is normalized to 1
    r = h * mu  # Radius of the base
    true_volume = (1 / 3) * np.pi * (r ** 2) * h  # Volume of a cone formula
    origin = np.zeros((1, 3))
    four_vector_with_origin = np.vstack((four_vector, origin))
    eight_vector_with_origin = np.vstack((eight_vector, origin))
    # print("four_vector_with_origin", four_vector_with_origin)
    # # Compute convex hull volumes
    four_vector_volume = ConvexHull(four_vector_with_origin).volume
    eight_vector_volume = ConvexHull(eight_vector_with_origin).volume

    ######################

    print('True volume:', np.round(true_volume, 4) if true_volume is not None else 'Not implemented')
    print('4 edge volume:', np.round(four_vector_volume, 4) if four_vector_volume is not None else 'Not implemented')
    print('8 edge volume:', np.round(eight_vector_volume, 4) if eight_vector_volume is not None else 'Not implemented')
    print()

def convex_hull(wrenches):
    """
    Given a set of wrenches, determine if the object is in force closure using a convex hull.
    :param wrenches: A numpy array of shape (NUM_FRICTION_CONE_VECTORS*n, 6) where n is the number of contact points.
    :return: None
    """

    if len(wrenches) == 0:
        print('No wrench input. Is calculate_friction_cone implemented?')

    #Convex Hull
    hull = None
    #Force closure boolean
    force_closure_bool = False
    #Radius of largest hypersphere contained within convex hull
    max_radius = 0.0

    ### YOUR CODE HERE ###
    try:
        # Compute the convex hull of the wrenches using the 'QJ' parameter to avoid numerical issues
        hull = ConvexHull(wrenches, qhull_options='QJ')

        # Check if the origin is inside the convex hull
        # The convex hull is defined as a set of half-space inequalities Ax + b <= 0, where A and b are given by hull.equations
        origin = np.zeros(wrenches.shape[1])  # The origin in 6D space
        # Check if the origin satisfies all the inequalities
        force_closure_bool = np.all(np.dot(hull.equations[:, :-1], origin) + hull.equations[:, -1] <= 0)

        if force_closure_bool:
            # Calculate the radius of the largest hypersphere centered at the origin
            # The radius is the shortest distance from the origin to each hyperplane: -b / ||A||
            distances = -hull.equations[:, -1] / np.linalg.norm(hull.equations[:, :-1], axis=1)
            max_radius = np.min(distances)
    except Exception as e:
        print(f"Error occurred while computing the convex hull: {e}")

        ### Your code ends here ###
    if hull is None:
        print('convex_hull function not implemented')
    else:
        if force_closure_bool:
            print('In force closure. Maximum hypersphere radius:', np.round(max_radius, 4))
        else:
            print('Not in force closure')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--g1', action='store_true', help='Run grasp 1')
    parser.add_argument('--g2', action='store_true', help='Run grasp 2')
    parser.add_argument('--custom', nargs='+', help='Run a custom grasp. Input is a list of 4 numbers: x, y, z, theta')
    args = parser.parse_args()

    world = World()
    
    print('\n\n\n========================================\n')
    input('Environment initialized. Press <ENTER> to execute grasp.')

    if args.g1:
        contact_points = world.grasp([.0, .08, 0.03, 0])
        print('\n========================================\n')
        input(f'Grasp 1 complete. Press <ENTER> to compute friction cone volumes and force closure.\n')
        print(f'Grasp 1 Contact point 1 Volumes:')
        compare_discretization(contact_points[0], world)

    if args.g2:
        contact_points = world.grasp([.0, .02, .01, 0])
        print('\n========================================\n')
        input(f'Grasp 2 complete. Press <ENTER> to compute force closure.\n')

    
    if args.custom:
        x, y, z, theta = args.custom
        contact_points = world.grasp([float(x), float(y), float(z), float(theta)])
        print('\n========================================\n')
        input(f'Custom grasp complete. Press <ENTER> to compute force closure.')

    wrenches = calculate_wrenches(contact_points, world)
    convex_hull(wrenches)
    print('\n\n')

if __name__ == '__main__':
    main()