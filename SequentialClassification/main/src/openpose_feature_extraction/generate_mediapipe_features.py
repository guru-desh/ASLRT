import cv2
import mediapipe as mp
import json
import math
from p_tqdm import p_imap
from functools import partial

'''
This function was already implemented in a different script .py file in this directory by someone else before Guru
The original script is called "mediapipePythonWrapper.py" and can be found in this directory: "/home/aslr/ProcessingPipeline/Scripts"
Edits were made to get non-normalized values and parallelize process
'''
def extract_mediapipe_features(frames, save_filepath, num_jobs, normalize_xy=True) -> None:
    """extract_mediapipe_features extracts the mediapipe features for each frame passed in as a list

    Arguments:
        frames {list} -- list of frames where each frame is a NumPy array with dtype uint8
        save_filepath {str} -- the location of where to save the resulting JSON features

    Keyword Arguments:
        normalize_xy {bool} -- whether to normalize the x, y coordinates (default: {True})
    """
    if frames == None:
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.1
        )
        return

    curr_frame = 0
    features = {}
    features_iterator = p_imap(partial(extract_frame_features, normalize_xy=normalize_xy), frames, num_cpus=num_jobs, disable=True)
    for curr_frame, curr_frame_feature in enumerate(features_iterator):
        features[curr_frame] = curr_frame_feature
  
    if save_filepath == None or save_filepath == 'None' or save_filepath == False or save_filepath.lower() == 'n':
        return features

    with open(save_filepath, "w") as outfile:
        json.dump(features, outfile, indent=4)
        
def extract_frame_features(image, normalize_xy) -> dict:
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.1
    )
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape
    image.flags.writeable = False
    results = holistic.process(image)
    curr_frame_features = {"pose": {}, "landmarks": {0: {}, 1: {}}}
    available_features = [results.left_hand_landmarks,
                            results.right_hand_landmarks, results.pose_landmarks]
    feature_location = [curr_frame_features["landmarks"][0],
                        curr_frame_features["landmarks"][1], curr_frame_features["pose"]]
    for index, curr_feature in enumerate(available_features):
        feature_num = 0
        if curr_feature is None:
            feature_location[index] = "None"
        else:
            for curr_point in curr_feature.landmark:
                if normalize_xy:
                    feature_location[index][feature_num] = [
                        curr_point.x, curr_point.y, curr_point.z
                    ]
                else:
                    feature_location[index][feature_num] = [
                        min(math.floor(curr_point.x * image_width), image_width - 1),
                        min(math.floor(curr_point.y * image_height), image_height - 1),
                    ]
                feature_num += 1
    holistic.close()
    
    return curr_frame_features
        