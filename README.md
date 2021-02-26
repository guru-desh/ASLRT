# CopyCat HTK Pipeline

A pipeline to run tracking Kinect, MediaPipe and AlphaPose experiments on sign language feature data using either HMMs, SBHMMs, or Transformers.

## Getting Started

* Navigate to either `projects/Kinect`, `projects/Mediapipe` or `projects/AlphaPose` depending on which hand tracking the experiment is for
* Inside this directory, navigate to `configs/features.json` to select which features to use. `"all_features"` is the complete list of features that are available for the specific hand tracker. Select a subset of those features to add into `"selected_features"` to customize the list of features to run the experiment on. Note that `config_kinect_features.py` and `config_alphapose_features.py` allows a faster way to print out which features you want to use for Kinect and AlphaPose instead of manually copying and pasting each one.
* Inside this directory, navigate to `configs/features.json` `"feature_dir"` to select where the base directory of the features directory is located. More likely than not, this will not be required to change on the current system. Note that for AlphaPose, you are required to run `test.py` on the Data in the `"feature_dir"` so that it aligns with the naming convention the pipeline uses.
* `configs/` also contains other HMM hyperparameters such as :
  * `hhed#.conf` which lists the number of mixture models. The current pipeline will iterate through these `hhed#.conf` files using the `train_iters` argument when running the experiment. Each `hhed#.conf` file has content in `MU <number of mixture models> {*.state[<state range start>-<state range end>].mix}`. Example: `MU 3 {*.state[2-29].mix}`.
  * `prototypes.json` contains a json indicating the number of states each word will have. You can change this json to fine tune the number of states. Each model (word) can have different number of states. 

## Running the Pipeline

* Open a terminal at `projects/Kinect`, `projects/Mediapipe` or `projects/AlphaPose` and execute (this is one specific example): `python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 180 200 220 --hmm_insertion_penalty 0 --cross_val_method leave_one_user_out --n_splits 4 --cv_parallel --parallel_jobs 5 --prepare_data` 
* This will first prepare the HTK files for HMMs on all users data located at the base directory `"features_dir"` using user independent cross validation of 4 folds on 5 parallel processes. The specifics of some common options are given in the next section. Please look at `main/src/main.py` for the complete list of arguments that can be used with the pipeline.

## HMM/SBHMM Pipeline input parameters

Note that the important ones are indicated with [**IMP**]

* General Parameters:
  
  * [**IMP**] `driver.py --prepare_data`: If the list of users or the type of features have changed, then it is required to set this flag to recompile the data files and generate ARK/HTK files. There is no harm in always using this flag if you are unsure.
  * `driver.py --save_results`: If you want to add results to the `all_results.json` file. Ideally, this file will contain all results from all experiments.
  * `driver.py --save_results_file`: Set to `all_results.json`. Don't change unless you know what you are doing. 

* Data List Parameters
  
  * [**IMP**] `driver.py --test_type`: The type of testing to perform `(none, test on train, cross validation, standard)`. cross validation ('cross_val') is the most likely option for generalized testing.
  * [**IMP**] `driver.py --users`: Specify a list of users (keywords) to run the visualization on a refined list of `.data` files. If empty, then all users in `"features_dir"` are used. Be careful with left hand vs right hand datasets.
  * [**IMP**] `driver.py --cross_val_method`: The type of split for cross validation (`kfold`, `leave one phrase out`, `stratified`,`leave one user out`, `user_dependent`). For user independent use `leave_one_user_out`. 
  * [**IMP**] `driver.py --n_splits`: The number of splits to perform in cross validation. For the current dataset `10` is normal (also the default value).
  * [**IMP**] `driver.py --cv_parallel`: `True` if you want to run `cross_val` folds in parallel. Useful for both `leave_on_user_out` and `user_dependent`.
  * [**IMP**] `driver.py --parallel_jobs`: If `cv_parallel` is set then this is the number of processes to use. It is recommeneded to make this equal to the number of users or splits you have. 
  * `driver.py --phrase_len`: Specify if you want to include only the phrases of a specific length in your dataset. Set to `0` (which is also the default value) if you want to include all phrases.
  * `driver.py --random_state`: Set to make sure splits don't change and results are the same everytime. Default value is set to `42`. Change only if you want to test different splits.

* HMM Training Parameters
  
  * [**IMP**] `driver.py --train_iters`: This is the iterations to perform on the model training. We are currently using something close to `3 25 50 75 100 120 140 160 180 200 220`. While you can change the values, remove items from the list, or add items to the list, it is unlikely this will deviate too much. Feel free to test with different values/size though. Note that this corresponds to the `hhed#.conf` files so make sure there is enough files for each element in this train iterations list.
  * [**IMP**] `driver.py --hmm_insertion_penalty`: This helps balance the percentage of insertions and deletions on the experiment. Ideally we want both of them to be around equal. A larger value will decrease the percentage of inserts performed.
  * `driver.py --mean`: The initial mean value used for training. Unlikely to change.
  * `driver.py --variance`: The initial variance value used for training. Unlikely to change.
  * `driver.py --transition_prob`: The initial transition probability value used for training. Unlikely to change.

* SBHMM Training Parameters

  * [**IMP**] `driver.py --train_sbhmm`: `True` if you want to train SBHMMs. Note that currently we are only using `SBHMMs` for recognition. Use `HMMs` for recognition
  * [**IMP**] `driver.py --sbhmm_iters`: The SBHMM equivalent of `train_iters`. For SBHMM, `3 25 50 75 100` has worked well.
  * [**IMP**] `driver.py --include_word_position`: `True` if you want to assign different label to the same word apearing in a different position. Including has helped results in the past. 
  * [**IMP**] `driver.py --include_word_level_states`: `True` if you want to include all the states of each model (word). Including has helped results in the past. 
  * [**IMP**] `driver.py --sbhmm_insertion_penalty`: SBHMM equivalent of `hmm_insertion_penalty`.
  * [**IMP**] `driver.py --classifier`: Which classifier you want to use for labelling states. Options are (`knn` - default, `adaboost`). `KNN` works better right now. We are adding more classifiers. 
  * [**IMP**] `driver.py --neighbors`: Number of neighbors used by KNN to classify label. 
  * `driver.py --sbhmm cycles`: Number of times you want to train SBHMMs. Increasing it can result in overfitting. Default value is `1`.
  * `driver.py --pca`: `True` if you want to PCA files produced by the classifier. So far, PCA'ing has only hurt our results. 
  * `driver.py --pca_components`: Number of PCA componenets that should be used if you are PCA'ing features after the classifier training.
  * `driver.py --multiple_classifiers`: Train 1 binary classifier for each label. Tends to hurt results. 
  * `driver.py --parallel_classifier_training`: Can be used to train classifiers in parallel if you are using `multiple_classifiers`. Tends to surprisingly slow down computation than just training them one by one. 
  * `driver.py --beam_threshold`: Threshold to prune branches while doing `HVITE`. Also used for verification. Don't change unless you know what you are doing. 

* Testing Parameters
  * [**IMP**] `driver.py --method`: Whether to perform `recognition` or `verification`.
  * `driver.py --start`: This determines where we start testing. Unlikely to change.
  * `driver.py --end`: This determines where we end testing. Unlikely to change.

## Transformer Pipeline input parameters

* The Transformer Pipeline is located in the TransformerClassification directory. Configuration details can be found in the configs subdirectory (currently, transformer_small.yaml is used). For further information, refer to [joey-nmt documentation](https://github.com/joeynmt/joeynmt).

* General Parameters:
  * [**IMP**] `mainDriver.py --test_type`: The type of testing to perform `(none, test on train, cross validation, standard)`. cross validation ('cross_val') is the most likely option for generalized testing.
  * [**IMP**] `mainDriver.py --users`: Specify a list of users (keywords) to run the visualization on a refined list of `.data` files. If empty, then all users in `"features_dir"` are used. Be careful with left hand vs right hand datasets.
  * [**IMP**] `mainDriver.py --cross_val_method`: The type of split for cross validation (`kfold`, `leave one phrase out`, `stratified`,`leave one user out`, `user_dependent`). For user independent use `leave_one_user_out`. 
  * [**IMP**] `mainDriver.py --n_splits`: The number of splits to perform in cross validation. For the current dataset `10` is normal (also the default value).
  * [**IMP**] `mainDriver.py --n_jobs`: The number of processes to run at once. For optimal performance, set it to the maximum number of threads on your system's CPU.
  * [**IMP**] `mainDriver.py --cv_parallel`: `True` if you want to run `cross_val` folds in parallel. Useful for both `leave_on_user_out` and `user_dependent`.
  * [**IMP**] `mainDriver.py --random_state`: Set to make sure splits don't change and results are the same everytime. Default value is set to `42`. Change only if you want to test different splits. 
  * [**IMP**] `mainDriver.py --parallel`: `True` if you want to parallelize the generation of features.
  * [**IMP**] `mainDriver.py --feature_extractor`: The feature to select. Generated in the data subdirectory by writeLists.py based on inputted type of data.

* Transformer Parameter:
  * `mainDriver.py --classifier`: The classifier for the transformer to use `(knn, adaboost)`

* KNN-specific Parameters:
  * `mainDriver.py --multiple_classifiers`
  * `mainDriver.py --knn_neighbors`
  * `mainDriver.py --pca_components`
  * `mainDriver.py --no_pca`


### Commands

* Prepare Data Only: `python3 driver.py --test_type none --prepare_data --users <user_list>`
* Recognition:
  * User-Independent- 
    * HMMs: `python3 driver.py --test_type cross_val --train_iters <iterations> --users <user_list> --cross_val_method leave_one_user_out --cv_parallel --parallel_jobs 12 --hmm_insertion_penalty -80`
    * SBHMMs: `python3 driver.py --test_type cross_val --train_iters <iterations> --sbhmm_iters <sbhmm_iterations> --users <user_list> --cross_val_method leave_one_user_out --cv_parallel --parallel_jobs 12 --hmm_insertion_penalty -80 --train_sbhmm --include_word_level_states --include_word_level_states --sbhmm_insertion_penalty -80 --neighbors 73`
  * User-Dependent-
    * HMMs: `python3 driver.py --test_type cross_val --train_iters <iterations> --users curr_user --cross_val_method kfold --n_splits 10 --cv_parallel --parallel_jobs 10 --hmm_insertion_penalty -80`
    * SBHMMs: `python3 driver.py --test_type cross_val --train_iters <iterations> --sbhmm_iters <sbhmm_iterations> --users curr_user --cross_val_method kfold --n_splits 10 --cv_parallel --parallel_jobs 10 --hmm_insertion_penalty -80 --train_sbhmm --include_word_level_states --include_word_level_states --sbhmm_insertion_penalty -80 --neighbors 73`
  * User-Adaptive-
    * HMMs: `python3 driver.py --test_type cross_val --train_iters <iterations> --users <user_list> --cross_val_method stratified --n_splits 10 --cv_parallel --parallel_jobs 10 --hmm_insertion_penalty -80`
    * SBHMMs: `python3 driver.py --test_type cross_val --train_iters <iterations> --sbhmm_iters <sbhmm_iterations> --users <user_list> --cross_val_method stratified --n_splits 10 --cv_parallel --parallel_jobs 10 --hmm_insertion_penalty -80 --train_sbhmm --include_word_level_states --include_word_level_states --sbhmm_insertion_penalty -80 --neighbors 73`
* Verification-
  * Zahoor (User Independent): `python3 driver.py --test_type cross_val --train_iters <iterations> --users <user_list> --cross_val_method leave_one_user_out --cv_parallel --parallel_jobs 12 --hmm_insertion_penalty -80 --method verification`
  * Simple-
    * Test on Train: `python3 driver.py --test_type test_on_train --train_iters <iterations> --users <user_list> --method verification`
    * Single Split: `python3 driver.py --test_type standard --train_iters <iterations> --users <user_list> --method verification`

Note that:
  * `user list`: `p1 p2 p3 p4 p5 p6 p7 p8`. 
  * `iterations`: `3 25 50 75 100 120 140 160 180 200 220`.
  * `sbhmm_iterations`: `3 25 50 75 100`.

# Segmentally Boosted HMMs

Extend the concept of the segmentally boosted hidden Markov models as described in [Pei Yin's disseration](https://smartech.gatech.edu/handle/1853/33939) to unsupervised sequential classification.

The sequence classification code is taken from [Here](https://github.com/lichard49/Segmentally-Boosted-HMMs). Datasets can also be found there. 

# Verification

* Simple Verfication: Refers to using HVITE for alignment and using a static threshold for accepting and rejecting phrases
* Verifcation Zahoor:  Refers to the verification algorithm proposed by Zahoor. You can find more information here: https://dl.acm.org/doi/pdf/10.1145/2070481.2070532

## Future Work

* Verification experiments on all 3 recognizers. 
* Extensive comparison with Transformers. 