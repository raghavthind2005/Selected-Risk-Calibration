import numpy as np

# NOTE: the discovered form evaluates sqrt(x5_n + 1e-8). SLSQP line
# search steps marginally outside the box, giving x5_n < -1e-8 and a NaN
# h_hat, which silently corrupts the solve. Clamped to the box-consistent
# branch. Inside the box (x5_n >= 0) this is bit-identical to the original.
def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Symbolic predictor h_hat(x) for an unknown 5-variable process constraint.

    Args:
        x      : shape (n_samples, 5).  Columns:
                   x[:,0] = x1  in [450, 1500]      (flow-rate parameter)
                   x[:,1] = x2  in [450, 1500]      (flow-rate parameter)
                   x[:,2] = x3  in [997.18, 1348.12] (thermal parameter)
                   x[:,3] = x4  in [0.5, 2.0]       (geometric parameter)
                   x[:,4] = x5  in [10, 100]        (geometric parameter)
        params : learnable constants (up to MAX_NPARAMS=12).

    Return:
        1-D array of h_hat values. Feasible region: h_hat(x) <= 0.

    Start simple (a normalised linear form is provided) and let the `fit_hint`
    diagnostics tell you which variables need nonlinear / interaction / ratio
    terms and which can be dropped. Not every input need appear.
    """
    x1, x2, x3, x4, x5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    # Normalise each variable to roughly [0, 1] for numerical stability.
    x1_n = (x1 - 450.0) / 1050.0
    x2_n = (x2 - 450.0) / 1050.0
    x3_n = (x3 - 997.18) / 350.94
    x4_n = (x4 - 0.5) / 1.5
    x5_n = (x5 - 10.0) / 90.0
    
    # Reduced linear contribution - lower coefficients for x1, x2, x3 to reduce over-conservatism
    linear = (params[0] * x1_n + params[1] * x2_n + params[2] * x3_n
              + params[3] * x4_n)
    
    # x5 with softer penalty
    x5_term = params[4] * np.sqrt(np.maximum(x5_n, 0.0) + 1e-8) + params[5] * x5_n
    
    # Reduced interactions
    flow_interaction = params[6] * x1_n * x2_n
    x3_flow_interaction = params[7] * x3_n * (x1_n + x2_n)
    geom_interaction = params[8] * x4_n * x5_n
    
    # Add negative bias to shift predictions down in feasible region
    # This directly addresses the over-conservative bias
    bias = params[9] - 2.5  # Shift down by ~2.5 to match the hint
    
    return linear + x5_term + flow_interaction + x3_flow_interaction + geom_interaction + bias

