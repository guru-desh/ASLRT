# dataset_diff.py
# Compares two datasets for differences in processed files.
# Usage: edit kinect_dir and mediapipe_dir, then run "python dataset_diff.py" (designed for Ubuntu 18.04 and Python 3)

import os, sys, glob

kinect_dir = '/path/to/Kinect/dataset/to/process'
mediapipe_dir = '/path/to/Mediapipe/dataset/to/process' #videos

kinect_filepaths = glob.glob(os.path.join(kinect_dir, '**'), recursive = True)
kinect_filepaths = list(filter(lambda filepath: os.path.isfile(filepath), kinect_filepaths))
kfps = []
for fp in kinect_filepaths:
    kfps.append('/'.join(fp.split('/')[-3:-1]))

mediapipe_filepaths = glob.glob(os.path.join(mediapipe_dir, '**'), recursive = True)
mediapipe_filepaths = list(filter(lambda filepath: os.path.isfile(filepath), mediapipe_filepaths))
mfps = []
for fp in mediapipe_filepaths:
    mfps.append('/'.join(fp.split('/')[-3:-1]))
    if '.mkv' not in fp: print("EXCEPTION: ", fp)


def Diff(list1, list2):
    return list(set(list(set(list1)-set(list2)) + list(set(list2)-set(list1))))

diffs = Diff(kfps, mfps)
phrases = {diff.split('/')[0] for diff in diffs}

print(len(diffs))
print(diffs[:5])
print(len(phrases))
print(sorted(phrases))
