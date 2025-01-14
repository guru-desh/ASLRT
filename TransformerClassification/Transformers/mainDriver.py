#CV = python3 mainDriver.py --test_type cross_val --cross_val_method stratified --n_splits 10

import sys
import glob
import random
import argparse
import os
import shutil
import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneGroupOut, train_test_split)
from statistics import mean
import joeynmt
import generator
from joeynmt.training import train
from generator.generateNewFeatures import generateFeatures


def returnUserDependentSplits(unique_users, htk_filepaths, test_size):
    splits = [[[],[]] for i in range(len(unique_users))]
    for htk_idx, curr_file in enumerate(htk_filepaths):
        curr_user = curr_file.split("/")[-1].split(".")[0].split('_')[-2]
        for usr_idx, usr in enumerate(unique_users):
            if usr == curr_user:
                if random.random() > test_size:
                    splits[usr_idx][0].append(htk_idx)
                else:
                    splits[usr_idx][1].append(htk_idx)
    splits = np.array(splits)
    return splits

def writeFiles(trainPaths, trainLabels, testPaths, testLabels):
    trainFiles = "\n".join(trainPaths)
    testFiles = "\n".join(testPaths)

    trainPathFile = open(f'../data/{args.feature_extractor}/lists/train.data', 'w')
    trainLabelFile = open(f'../data/{args.feature_extractor}/lists/train.en', 'w')
    devPathFile = open(f'../data/{args.feature_extractor}/lists/dev.data', 'w')
    devLabelFile = open(f'../data/{args.feature_extractor}/lists/dev.en', 'w')
    testPathFile = open(f'../data/{args.feature_extractor}/lists/test.data', 'w')
    testLabelFile = open(f'../data/{args.feature_extractor}/lists/test.en', 'w')

    trainPathFile.write(trainFiles)
    devPathFile.write(testFiles)
    testPathFile.write(testFiles)

    trainLabelFile.writelines(trainLabels)
    devLabelFile.writelines(testLabels)
    testLabelFile.writelines(testLabels)


def getLabels(ark_paths: list):
    labels = [" ".join(i.split("/")[-1].split(".")[1].split("_"))+"\n" for i in ark_paths]
    return labels

def getUsers(ark_paths: list):
    users = [filepath.split('/')[-1].split('.')[0].split('_')[-2] for filepath in ark_paths]
    return users

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("SignJoey")
    ######################### ARGUMENTS #############################
    parser.add_argument('--test_type', type=str, default='test_on_train',
                        choices=['test_on_train', 'cross_val', 'standard'])
    parser.add_argument('--users', nargs='*', default=[])
    parser.add_argument('--phrase_len', type=int, default=0)
    parser.add_argument('--random_state', type=int, default=24)
    parser.add_argument('--cross_val_method', required='cross_val' in sys.argv,
                        default='kfold', choices=['kfold',
                                                  'leave_one_phrase_out',
                                                  'stratified',
                                                  'leave_one_user_out',
                                                  'user_dependent'])
    parser.add_argument('--n_splits', required='cross_val' in sys.argv,
                        type=int, default=10)
    parser.add_argument('--transform_files', action='store_true')
    parser.add_argument('--create_transform_files', action='store_true')
    parser.add_argument('--cv_parallel', action='store_true')
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument("--config_path", type=str, help="path to YAML config file")
    parser.add_argument('--classifier', type=str, default='knn',
                        choices=['knn', 'adaboost'])
    parser.add_argument('--include_state', action='store_true')
    parser.add_argument('--include_index', action='store_true')
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--multiple_classifiers', action='store_true')
    parser.add_argument('--knn_neighbors', default=70)
    parser.add_argument('--pca_components', default=92)
    parser.add_argument('--no_pca', action='store_true')
    parser.add_argument('--feature_extractor', default='mediapipe')    
    args = parser.parse_args()
    ###################################################################

    print("Version 0.0.4")

    cross_val_methods = {'kfold': (KFold, False),
                         'leave_one_phrase_out': (LeaveOneGroupOut(), True),
                         'stratified': (StratifiedKFold, True),
                         'leave_one_user_out': (LeaveOneGroupOut(), True),
                         'user_dependent': (None, True)
    }
    cross_val_method, use_groups = cross_val_methods[args.cross_val_method]

    if len(args.users) == 0:
        ark_filepaths = glob.glob(f'../data/{args.feature_extractor}/ark/*ark')
        random.shuffle(ark_filepaths)
    else:
        ark_filepaths = []
        for user in args.users:
            ark_filepaths.extend(glob.glob(os.path.join(f"../data/{args.feature_extractor}/ark", '*{}*.ark'.format(user))))
    
    ark_labels = getLabels(ark_filepaths)
    dataset_users = getUsers(ark_filepaths)

    if args.test_type == 'test_on_train':
        
        train_paths = ark_filepaths
        train_labels = ark_labels
        test_paths = ark_filepaths
        test_labels = ark_labels

        print(f'Nmber of elements in train_paths = {str(len(train_paths))}')
        print(f'Nmber of elements in test_paths = {str(len(test_paths))}')

        writeFiles(train_paths, train_labels, test_paths, test_labels)
        train(args.config_path)

    
    if args.test_type == 'standard':

        train_paths, test_paths, train_labels, test_labels = train_test_split(
            ark_filepaths, ark_labels, test_size=args.test_size,
            random_state=args.random_state)

        print(f'Nmber of elements in train_paths = {str(len(train_paths))}')
        print(f'Nmber of elements in test_paths = {str(len(test_paths))}')

        writeFiles(train_paths, train_labels, test_paths, test_labels)
        train(args.config_path)
    
    if args.test_type == 'cross_val' and args.transform_files:

        unique_users = set(dataset_users)
        group_map = {user: i for i, user in enumerate(unique_users)}
        groups = [group_map[user] for user in dataset_users]            
        cross_val = cross_val_method
        splits = list(cross_val.split(ark_filepaths, ark_labels, groups))
        
        all_results = []
        user_order = []
        
        for i, (train_index, test_index) in enumerate(splits):

            print(f'Current split = {i}')
            
            train_paths = np.array(ark_filepaths)[train_index]
            train_labels = np.array(ark_labels)[train_index]
            test_paths = np.array(ark_filepaths)[test_index]
            test_labels = np.array(ark_labels)[test_index]
            curr_user = getUsers(test_paths)[0]
            user_order.append(curr_user)

            print(f'Current user = {curr_user}')

            curr_alignment_file = glob.glob(f'../data/{args.feature_extractor}/alignment/{curr_user}/*.mlf')[-1]

            if args.create_transform_files:

                print(f'Starting feature generation')

                generateFeatures(curr_alignment_file, f"../data/{args.feature_extractor}/ark/", classifier=args.classifier, include_state=args.include_state, 
                            include_index=args.include_index, n_jobs=args.n_jobs, parallel=args.parallel, trainMultipleClassifiers=args.multiple_classifiers,
                            knn_neighbors=int(args.knn_neighbors), generated_features_folder=f'../data/{args.feature_extractor}/transformed/{curr_user}/', 
                            pca_components=args.pca_components, no_pca=args.no_pca)
            else: 
                transformedFiles = glob.glob(f'../data/{args.feature_extractor}/transformed/{curr_user}/*.ark')
                train_paths = []
                train_labels = []
                test_paths = []
                test_labels = []
                for filePath in transformedFiles:
                    if curr_user in filePath.split("/")[-1]:
                        test_paths.append(filePath)
                        test_labels.append(getLabels([filePath])[0])
                    else:
                        train_paths.append(filePath)
                        train_labels.append(getLabels([filePath])[0])
                
                print(f'Number of elements in train_paths = {str(len(train_paths))}')
                print(f'Number of elements in test_paths = {str(len(test_paths))}')
                
                writeFiles(train_paths, train_labels, test_paths, test_labels)
                all_results.append(train(args.config_path))    
            

    elif args.test_type == 'cross_val':
        unique_users = []
        if args.cross_val_method == 'leave_one_user_out' or args.cross_val_method == 'user_dependent':
            unique_users = set(dataset_users)
            group_map = {user: i for i, user in enumerate(unique_users)}
            groups = [group_map[user] for user in dataset_users]            
            cross_val = cross_val_method
        else:
            unique_phrases = set(ark_labels)
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[label] for label in ark_labels]
            cross_val = cross_val_method(n_splits=args.n_splits)
        
        if args.cross_val_method == 'user_dependent':
            splits = returnUserDependentSplits(unique_users, ark_filepaths, args.test_size)
        elif use_groups:
            splits = list(cross_val.split(ark_filepaths, ark_labels, groups))
        else:
            splits = list(cross_val.split(ark_filepaths, ark_labels))
        
        all_results = []
        user_order = []
        
        for i, (train_index, test_index) in enumerate(splits):

            print(f'Current split = {i}')
            
            train_paths = np.array(ark_filepaths)[train_index]
            train_labels = np.array(ark_labels)[train_index]
            test_paths = np.array(ark_filepaths)[test_index]
            test_labels = np.array(ark_labels)[test_index]
            users = getUsers(test_paths)
            user_order.append(users[0])

            print(f'Nmber of elements in train_paths = {str(train_paths.shape)}')
            print(f'Nmber of elements in test_paths = {str(test_paths.shape)}')
            print(f'Current user = {users[0]}')
            print(f'Test Users = {users}')

            writeFiles(train_paths, train_labels, test_paths, test_labels)
            print(args.config_path)
            curr_result = train(args.config_path)
            print("Current Fold result = ")
            print(curr_result)
            all_results.append(curr_result)
        
        all_results = np.array(all_results)
        print(f'All results = {str(all_results)}')
        average_results = np.mean(all_results, axis=0)
        print(f'Cross validation results = {str(average_results)}')
        print(f'User order = {str(user_order)}')





