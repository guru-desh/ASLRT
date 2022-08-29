import numpy as np
from numba import njit
from .data_augmentation_utils import *

@njit(cache=True, parallel=True)
def get3DMediapipeCoordinates(video, num_jobs) -> list:
    """get3DMediapipeCoordinates get the mediapipe coordinates (non-normalized) and their actual depth according to the Azure Kinect Depth Camera

    Arguments:
        videos {list} -- list of strings containing the video paths

    Returns:
        list -- all the pose points (represented by a Nx3 NumPy Array) for each 
    """
    # Open the depth and the color videos
    cameraIntrinsicMatrix = getCameraIntrinsicMatrix(video)
    colorFrames = getColorFrames(video)
    depthFrames = getDepthFrames(video)
    videoFrames = np.load(f"{video}.npz")

    currVideo = []

    for frame, depth in zip(colorFrames, depthFrames):

        frame = videoFrames[frame]
        depth = videoFrames[depth]

        currFrame = []

        # Get the openpose data in NumPy arrays
        currMediapipeFeatures = extract_mediapipe_features(
            [frame], normalize_xy=False, save_filepath=False, num_jobs=num_jobs)
        hand0Features = np.array(
            list(currMediapipeFeatures[0]['landmarks'][0].values()))
        hand1Features = np.array(
            list(currMediapipeFeatures[0]['landmarks'][1].values()))
        poseFeatures = np.array(
            list(currMediapipeFeatures[0]['pose'].values()))

        # Only add to currFrame if the hand and body are detected
        if len(hand0Features) > 0:
            currFrame.append(hand0Features)
        if len(hand1Features) > 0:
            currFrame.append(hand1Features)
        if len(poseFeatures) > 0:
            currFrame.append(poseFeatures)

        if len(currFrame) < 1:
            continue

        currFrame = np.vstack(currFrame)

        # Get the depth of each point, convert depth values to meters, and add the depth values back to currVideoFrames++
        # TODO: Find why the points are negative
        depthValues = np.apply_along_axis(lambda point: getNonZeroDepth(
            abs(int(point[1])), abs(int(point[0])), depth), 1, currFrame).reshape(-1, 1)
        depthValues = depthValues / 1000
        currFrame = np.hstack((currFrame, depthValues))

        # Get the world coordinates
        currFrame = calculateWorldCoordinates(
            currFrame, cameraIntrinsicMatrix)

        # Add this to the list containing all the frames of the current video
        currVideo.append(currFrame)

    currVideo = np.vstack(currVideo)

    return currVideo, cameraIntrinsicMatrix

@njit(cache=True, parallel=True)
def getNonZeroDepth(row, col, depth) -> float:
    """getNonZeroDepth gets the non-zero depth value of a point
    This approach differs from the "cleanDepthMap" method. This method works on cleaning the depth for one point. cleanDepthMap cleans up the whole entire depth map. 
    This method works best for just a single point since doing this process for the entire depth map is computationally expensive.
    This method is also supposed to only handle cases where the pose coordinate has a 0-depth voxel, which is considered as "invalid" by the Azure Kinect SDK.

    Arguments:
        row {int} -- the row index of the point
        col {int} -- the column index of the point
        depth {np.ndarray} -- the depth map of the frame

    Returns:
        float -- the non-zero depth at a specific pixel
    """
    if depth[row, col] != 0:
        return depth[row, col]

    # Initial values
    percentNonZero = 0
    radius = 0
    # We want the area considered to have a majority of non-zero depth values
    # We keep increasing the radius until we have a majority of non-zero depth values
    while percentNonZero < 0.5 and radius < depth.shape[0] and radius < depth.shape[1]:
        minRow = row - radius if row - radius > 0 else 0
        maxRow = row + radius if row + \
            radius < depth.shape[0] else depth.shape[0]
        minCol = col - radius if col - radius > 0 else 0
        maxCol = col + radius if col + \
            radius < depth.shape[1] else depth.shape[1]

        surroundingPoints = depth[minRow:maxRow, minCol:maxCol]
        percentNonZero = np.count_nonzero(
            surroundingPoints) / ((maxRow - minRow + 1) * (maxCol - minCol + 1))
        
        radius += 1

    # Consider the points where the depth is not zero
    nonZeroPoints = surroundingPoints[surroundingPoints != 0]

    # Return the average of the non-zero points
    return np.mean(nonZeroPoints)

@njit(cache=True, parallel=True)
def rotation_v_0(X_int, Y_int, Z_int, cameraIntrinsicMatrix) -> float:
    """rotation_v_0 calculates the minimum rotation going left

    Arguments:
        X_int {float} -- the world X coordinate of the point
        Y_int {float} -- the world Y coordinate of the point
        Z_int {float} -- the world Z coordinate of the point
        cameraIntrinsicMatrix {np.ndarray} -- 3x3 camera matrix

    Returns:
        float -- minimum rotation going left
    """
    f_y = cameraIntrinsicMatrix[1, 1]
    c_y = cameraIntrinsicMatrix[1, 2]

    numerator = (Z_int * c_y / f_y) + Y_int
    denominator = (-1 * Y_int * c_y / f_y) + Z_int
    theta_x_radians = np.arctan(numerator / denominator)
    theta_x_degrees = np.rad2deg(theta_x_radians)

    return theta_x_degrees

@njit(cache=True, parallel=True)
def rotation_v_2160(X_int, Y_int, Z_int, cameraIntrinsicMatrix) -> float:
    """rotation_v_2160 calculates the minimum rotation going right

    Arguments:
        X_int {float} -- the world X coordinate of the point
        Y_int {float} -- the world Y coordinate of the point
        Z_int {float} -- the world Z coordinate of the point
        cameraIntrinsicMatrix {np.ndarray} -- 3x3 camera matrix

    Returns:
        float -- minimum rotation going right
    """
    f_y = cameraIntrinsicMatrix[1, 1]
    c_y = cameraIntrinsicMatrix[1, 2]

    numerator = (-1 * Z_int * (2160 - c_y) / f_y) + Y_int
    denominator = (Y_int * (2160 - c_y) / f_y) + Z_int
    theta_x_radians = np.arctan(numerator / denominator)
    theta_x_degrees = np.rad2deg(theta_x_radians)

    return theta_x_degrees

@njit(cache=True, parallel=True)
def rotation_u_0(X_int, Y_int, Z_int, cameraIntrinsicMatrix) -> float:
    """rotation_u_0 calculates the minimum rotation going up

    Arguments:
        X_int {float} -- the world X coordinate of the point
        Y_int {float} -- the world Y coordinate of the point
        Z_int {float} -- the world Z coordinate of the point
        cameraIntrinsicMatrix {np.ndarray} -- 3x3 camera matrix

    Returns:
        float -- minimum rotation going up
    """
    f_x = cameraIntrinsicMatrix[0, 0]
    c_x = cameraIntrinsicMatrix[0, 2]

    numerator = (-1 * Z_int * (0 - c_x) / f_x) + X_int
    denominator = (X_int * (0 - c_x) / f_x) + Z_int
    theta_x_radians = np.arctan(numerator / denominator)
    theta_x_degrees = np.rad2deg(theta_x_radians)

    return theta_x_degrees

@njit(cache=True, parallel=True)
def rotation_u_3840(X_int, Y_int, Z_int, cameraIntrinsicMatrix) -> float:
    """rotation_u_3840 calculates the minimum rotation going down

    Arguments:
        X_int {float} -- the world X coordinate of the point
        Y_int {float} -- the world Y coordinate of the point
        Z_int {float} -- the world Z coordinate of the point
        cameraIntrinsicMatrix {np.ndarray} -- 3x3 camera matrix

    Returns:
        float -- minimum rotation going down
    """
    f_x = cameraIntrinsicMatrix[0, 0]
    c_x = cameraIntrinsicMatrix[0, 2]

    numerator = (-1 * Z_int * (2160 - c_x) / f_x) + X_int
    denominator = (X_int * (2160 - c_x) / f_x) + Z_int
    theta_x_radians = np.arctan(numerator / denominator)
    theta_x_degrees = np.rad2deg(theta_x_radians)

    return theta_x_degrees