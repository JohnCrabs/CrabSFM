#!/usr/bin/python

import numpy as np
from math import sqrt

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    centroid_A = [[centroid_A[0]], [centroid_A[1]], [centroid_A[2]]]
    centroid_B = [[centroid_B[0]], [centroid_B[1]], [centroid_B[2]]]

    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))

    # dot is matrix multiplication for array
    H = np.dot(Am, np.transpose(Bm))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n")
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    #print(R.shape)
    # X = Rx + t => t = X - Rx

    t = centroid_B + np.dot(-R, centroid_A)

    return R, t


"""
Example

# Test with random data

# Random rotation and translation
R = mat(random.rand(3,3))
t = mat(random.rand(3,1))

# make R a proper rotation matrix, force orthonormal
U, S, Vt = linalg.svd(R)
R = U*Vt

# remove reflection
if linalg.det(R) < 0:
   Vt[2,:] *= -1
   R = U*Vt

# number of points
n = 10

A = mat(random.rand(3, n));
B = R*A + tile(t, (1, n))

# Recover R and t
ret_R, ret_t = rigid_transform_3D(A, B)

# Compare the recovered R and t with the original
B2 = (ret_R*A) + tile(ret_t, (1, n))

# Find the root mean squared error
err = B2 - B
err = multiply(err, err)
err = sum(err)
rmse = sqrt(err/n);

print("Points A")
print(A)
print("")

print("Points B")
print(B)
print("")

print("Ground truth rotation")
print(R)

print("Recovered rotation")
print(ret_R)
print("")

print("Ground truth translation")
print(t)

print("Recovered translation")
print(ret_t)
print("")

print("RMSE:", rmse)

if rmse < 1e-5:
    print("Everything looks good!\n");
else:
    print("Hmm something doesn't look right ...\n");
"""
