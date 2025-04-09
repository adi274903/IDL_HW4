import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        Z_shifted = np.moveaxis(Z, self.dim, -1)
        original_shape = Z_shifted.shape
        flattened_Z = Z_shifted.reshape(-1, original_shape[-1])
        
        # Compute the softmax in a numerically stable way
        max_vals = np.max(flattened_Z, axis=1, keepdims=True)
        exp_Z = np.exp(flattened_Z - max_vals)
        sum_exp = np.sum(exp_Z, axis=1, keepdims=True)
        A_flat = exp_Z / sum_exp
        
        # Restore original dimensions
        self.A = np.moveaxis(A_flat.reshape(original_shape), -1, self.dim)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            A_shifted = np.moveaxis(self.A, self.dim, -1)
            dLdA_shifted = np.moveaxis(dLdA, self.dim, -1)
            batch_shape = A_shifted.shape[:-1]
            
            A_flat = A_shifted.reshape(-1, C)
            dLdA_flat = dLdA_shifted.reshape(-1, C)
            
        sum_terms = np.sum(A_flat * dLdA_flat, axis=1, keepdims=True)
        dLdZ_flat = A_flat * (dLdA_flat - sum_terms)

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            dLdZ_moved = dLdZ_flat.reshape(*batch_shape, C)
            dLdZ = np.moveaxis(dLdZ_moved, -1, self.dim)
        
            
        return dLdZ