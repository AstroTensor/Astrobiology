# normalize_rotations.py

import numpy as np
from directional_equations import schwarzschild_radius, planck_energy, hawking_temperature

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def exponential(x):
    return np.exp(x)

def cube_root(x):
    return np.cbrt(x)

def transformations(values):
    """
    Apply a series of transformations to the input values.
    """
    values = np.array(values)

    # Apply the sigmoid transformation
    sigmoid_transformed = sigmoid(values)

    # Apply the exponential transformation
    exponential_transformed = exponential(sigmoid_transformed)

    # Apply the cube root transformation
    cube_root_transformed = cube_root(exponential_transformed)

    return cube_root_transformed

def normalize_rotation_angles(angles):
    """
    Normalize rotation angles using astrophysical transformations.
    """
    # Convert angles to radians
    angles_rad = np.radians(angles)
    
    # Apply  transformations
    transformed_angles = transformations(angles_rad)
    
    # Apply additional  astrophysical equations
    schwarzschild_transformed = schwarzschild_radius(transformed_angles)
    planck_transformed = planck_energy(transformed_angles)
    hawking_transformed = hawking_temperature(transformed_angles)
    
    # Combine all transformations into a final normalization
    normalized_angles = (schwarzschild_transformed + planck_transformed + hawking_transformed) / 3
    
    return normalized_angles

def normalize_rotation_matrix(matrix):
    """
    Normalize a rotation matrix using astrophysical transformations.
    """
    # Ensure the input is a numpy array
    matrix = np.array(matrix)
    
    # Normalize each element in the rotation matrix
    normalized_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Apply transformations to each element
            transformed_value = transformations(matrix[i, j])
            schwarzschild_transformed = schwarzschild_radius(transformed_value)
            planck_transformed = planck_energy(transformed_value)
            hawking_transformed = hawking_temperature(transformed_value)
            
            # Combine all transformations
            normalized_value = (schwarzschild_transformed + planck_transformed + hawking_transformed) / 3
            normalized_matrix[i, j] = normalized_value
    
    return normalized_matrix

# FOR TESTS
# if __name__ == "__main__":
#     sample_angles = [0, 45, 90, 135, 180]
#     normalized_angles = normalize_rotation_angles(sample_angles)
#     print("Original angles:", sample_angles)
#     print("Normalized angles:", normalized_angles)
    
#     sample_matrix = [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ]
#     normalized_matrix = normalize_rotation_matrix(sample_matrix)
#     print("Original matrix:")
#     print(np.array(sample_matrix))
#     print("Normalized matrix:")
#     print(normalized_matrix)
