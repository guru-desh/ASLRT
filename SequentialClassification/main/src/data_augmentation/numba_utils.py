import numpy as np
from numba import njit

@njit(cache=True, parallel=True)
def asType(arr, dtype):
    return arr.astype(dtype)

@njit(cache=True, parallel=True)
def roundArray(x):
    return np.rint(x)

@njit(cache=True)
def applyMask(array, mask):
    return array[mask]

@njit(cache=True)
def copyThreeDPoints(threeDPoints):
    return np.copy(threeDPoints[:, -1]).reshape(-1, 1)

@njit(cache=True, parallel=True)
def homogenize3DCoordinates(projectedImage):
    projectedImage[0, :] = projectedImage[0, :] / projectedImage[2, :]
    projectedImage[1, :] = projectedImage[1, :] / projectedImage[2, :]
    return projectedImage
    
@njit(cache=True, parallel=True)
def applyDistortion(projectedImage, k1, k2, k3, k4, k5, k6, p1, p2, fx, cx, fy, cy):
    rSquared = projectedImage[0, :] ** 2 + projectedImage[1, :] ** 2
    distortionNumerator = 1 + k1 * rSquared + k2 * rSquared ** 2 + k3 * rSquared ** 3
    distortionDenominator = 1 + k4 * rSquared + k5 * rSquared ** 2 + k6 * rSquared ** 3
    projectedImage[0, :] = cx + fx * (distortionNumerator * projectedImage[0, :] / distortionDenominator) + (2 * p1 * projectedImage[0, :] * projectedImage[1, :]) + p2
    projectedImage[1, :] = cy + fy * (distortionNumerator * projectedImage[1, :] / distortionDenominator) + (p1 * (rSquared + 2 * projectedImage[1, :] ** 2)) + (2 * p2 * projectedImage[0, :] * projectedImage[1, :])
    return projectedImage