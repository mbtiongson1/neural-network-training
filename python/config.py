# Configuration for Neural Networks

NetworkA = {
    'methods'    : [1, 1, 0],  # [i, j, k] — 0 logistic, 1 tanh, 2 relu
    'a_l'        : 1.0,        # logistic slope
    'a_tanh'     : 1.716,      # tanh a
    'b_tanh'     : 0.66666,    # tanh b
    'a_relu'     : 0.01,       # leaky relu gamma
    'eta'        : 0.85,       # learning rate
    'alpha'      : 0.9,        # momentum constant
    'size'       : 8,          # hidden layer size
    'batch_size' : 8,          # mini-batch size
}

NetworkB = {
    'methods'    : [2, 2, 0],  # [i, j, k] — 0 logistic, 1 tanh, 2 relu
    'a_l'        : 1.0,        # logistic slope
    'a_tanh'     : 1.716,      # tanh a
    'b_tanh'     : 0.66666,    # tanh b
    'a_relu'     : 0.01,       # leaky relu gamma
    'eta'        : 0.85,        # learning rate
    'alpha'      : 0.9,        # momentum constant
    'size'       : 8,          # hidden layer size
    'batch_size' : 8,          # mini-batch size
}

# Custom Network configuration equivalent to NetworkB parameters
# Edit these dicts to experiment with different hyperparameters or activation functions
NetworkC = {
    'methods'    : [2, 2, 0],  # [i, j, k] — 0 logistic, 1 tanh, 2 relu
    'a_l'        : 1.0,        # logistic slope
    'a_tanh'     : 1.716,      # tanh a
    'b_tanh'     : 0.66666,    # tanh b
    'a_relu'     : 0.01,       # leaky relu gamma
    'eta'        : 0.01,        # learning rate
    'alpha'      : 0.8,        # momentum constant
    'size'       : 20,          # hidden layer size
    'batch_size' : 8,          # mini-batch size
}
