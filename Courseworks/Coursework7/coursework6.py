def load_data():
    
    import numpy as np
    
    """Load data and convert it to the metric system."""
    path_dataset = "height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454
    return height, weight, gender

def polynomial_basis(inputs, degree):  
    
    import numpy as np
    
    basis_matrix = np.ones((len(inputs), 1))
    for counter in range(1, degree + 1):
        basis_matrix = np.c_[basis_matrix, np.power(inputs, counter)]
    return basis_matrix

def regression(data_matrix, data_samples):
    
    import numpy as np
    
    gram_matrix = data_matrix.T @ data_matrix
    right_hand_side = data_matrix.T @ data_samples
    return np.linalg.solve(gram_matrix, right_hand_side)

def soft_thresholding(argument, threshold):
    
    import numpy as np
    
    return np.sign(argument) * np.maximum(0, np.abs(argument) - threshold)

def mean_squared_error(data_matrix, weights, outputs):
    
    import numpy as np
    
    return np.mean((data_matrix @ weights - outputs) ** 2) / 2

def lasso_cost_function(data_matrix, weights, outputs, regularisation_parameter):
    
    import numpy as np
    
    return mean_squared_error(data_matrix, weights, outputs) + regularisation_parameter \
            * np.sum(np.abs(weights))
