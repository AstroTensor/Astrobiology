import numpy as np
from directional_equations import schwarzschild_radius, planck_energy, hawking_temperature

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def exponential(x):
    return np.exp(x)

def cube_root(x):
    return np.cbrt(x)

def rescale(values):
    """
    Rescale a list of floats for reward normalization.
    """
    # Convert the list to a numpy array for vectorized operations
    values = np.array(values)
    
    # Apply the sigmoid transformation
    sigmoid_transformed = sigmoid(values)
    
    # Apply the exponential transformation
    exponential_transformed = exponential(sigmoid_transformed)
    
    # Apply the cube root transformation
    cube_root_transformed = cube_root(exponential_transformed)
    
    # Combining with complex equations from directional_equations
    schwarzschild_transformed = schwarzschild_radius(cube_root_transformed)
    planck_transformed = planck_energy(cube_root_transformed)
    hawking_transformed = hawking_temperature(cube_root_transformed)
    
    # Combine all transformations into a final rescaling
    combined_transformed = (schwarzschild_transformed + planck_transformed + hawking_transformed) / 3
    
    return combined_transformed

# For Testing
# if __name__ == "__main__":
    # sample_values = [0.5, 1.0, 1.5, 2.0, 2.5]

    # rescaled_values = rescale(sample_values)
    # print("Original values:", sample_values)
    # print("Rescaled values:", rescaled_values)
