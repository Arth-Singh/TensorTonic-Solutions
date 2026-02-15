import numpy as np 


def hinge_loss(y_true, y_score , margin =1.0, reduction ="mean"):

  y_true = np.array(y_true, dtype= np.float64)
  y_score = np.array(y_score, dtype = np.float64)


  if y_true.shape != y_score.shape:
    return None 


  if not np.isin(y_true, [-1,1]).all():
    return None


  product = y_true * y_score


  losses = np.maximum(0, margin - product)

  if reduction =="sum":
    return float(np.sum(losses))
  else: 
    return float(np.mean(losses))


    