import numpy as np

NUM_FRICTION_CONE_VECTORS = 4

def process_contact_point(contact_point):
    contact_pos = contact_point[6]
    contact_normal = contact_point[7]
    contact_force_scale = contact_point[9]
    tangent_dir = contact_point[11]

    #Pybullet gives tuples. Convert to numpy arrays
    contact_pos = np.array(contact_pos)
    contact_normal = np.array(contact_normal)
    tangent_dir = np.array(tangent_dir)

    contact_force_vector = np.array(contact_normal) * contact_force_scale
    return contact_pos, contact_force_vector, tangent_dir

def get_f0_pre_rotation(contact_force_vector, mu):
    normal_force_magnitude = np.linalg.norm(contact_force_vector)
    contact_unit_normal = contact_force_vector / normal_force_magnitude

    #Scale the friction cone discretization so that the cone radius is the same as the friction force
    cone_vector_length = np.sqrt(normal_force_magnitude ** 2 + (mu * normal_force_magnitude)**2)
    f_0_pre_rotation = cone_vector_length * contact_unit_normal
    return cone_vector_length, contact_unit_normal, f_0_pre_rotation

