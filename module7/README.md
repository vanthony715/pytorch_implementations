Step 0: Download and Unzip Dataset.
        Download UCF50 dataset from https://www.crcv.ucf.edu/data/UCF50.php .
        Under "UCF50 - Action Recognition Data Set: The data set can be downloaded" section, left click on the hyperling "clicking here". Unzip the rar files and save the dataset as follows:

        - UCF50
            - Action_Class1
            - Action_Class2
            ... ... ... ...
            - Action_Class51

Step 1: Preprocess the Video Dataset and Extract Frames from Each Video.
        Edit and run the bash script file preprocessing.sh to preprocess the extracted video dataset. In particular,

        --dataset_dir: the path where you saved the extracted dataset in Step 0.
        --frame_dir: the path where you will be saving the extracted frames from each video clip.
        --n_frames: a hyper-parameter which specifies how many frames you will sample randomly from each video. Feel free to use a different number.

Step 2: Run balance_dataset.py to remove samples of classes that have too much or too little data.
        Change the "datapath" to point to the UCF50 dataset.

Step 2: Run Training.
        Run the bash script file train.sh. You must change the "--frame_dir" argument to match where you store the extracted frames from Step 1. You may also play with other parameters to see how they affect the experiment results.

Step 3: Run Testing.
        Run the bash script file test.sh. You must change the "--ckpt" argument to match where you store the best model from step 2. 