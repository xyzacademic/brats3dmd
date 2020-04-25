Instruction
-

`single_train.py` - training script 

`single_validation.py` - validation script (if you can provide label)

`single_test.py` - inference script (output nii.gz file)

`lossfunction.py` - several useful loss function defined here

`segmenter.py` - segmentation class, in which defined training, testing, 
inference code and fp32 fp16 manager

`dataloader.py` - pytorch customize dataloader defined here, you can
modify it to satisfied your case.

`job` - a training and validation command and related args example. (focus on format)

`version` - several network structure variations defined here, v15 is the best one maybe.

`easylab` - (not finished) save some useful function and class.