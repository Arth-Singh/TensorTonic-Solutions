import numpy as np 

def cosine_embedding_loss(x1, x2, label, margin = 0.0):

    x1 = np.array(x1, dtype = np.float64)
    x2 = np.array(x2, dtype = np.float64)


    dot_product = np.dot(x1, x2)

    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)  # linalg computed L2 norm 

    #cosing loss

    cosine_sim = dot_product / (norm_x1 * norm_x2)


    if label == 1:
        loss = 1.00 - cosine_sim

    else:

        loss = max(0.0, cosine_sim - margin)

    return float(loss)
    