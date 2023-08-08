import numpy as np

def hypo(X, theta):
  return X.dot(theta)


def cost(X, y, theta, m):
  v = 1 / (2*m) * np.sum(np.square(X.dot(theta) - y))
  return v
  
  
X = np.array(
    [[ 2,  1],
    [ 7,  1],
    [ 9,  1],
    [ 3,  1],
    [10,  1],
    [ 6,  1],
    [ 1,  1],
    [ 8,  1]]
)
            
y = np.array(
    [[13],
    [35],
    [41],
    [19],
    [45],
    [28],
    [10],
    [55]]
    )

theta = np.array(
    [
        [1.],
        [1.]
    ])
            
m = 8.

print("Loss = {}".format(cost(X, y, theta, m)))