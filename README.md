This repo stores our progress for developing a transformer model to predict Python "if" conditions.

**Training Instructions**

To train the model yourself, clone the repository, replace dataset_filepath in model_training_c.py with the path to your desired training dataset and run ```python model_training_c.py```.
Having a high-end graphics card and plenty of RAM is recommended for efficiency and avoiding memory overflow during training.

**Testing Instructions**

When training is complete, there should be 2 folders with separate versions of the model: 1 after the pre-training step, and 1 after finetuning. To test your model, run ```python model_testing.py```.
If your testing machine does not have a NVIDIA graphics card, replace all instances of ".cuda()" with ".cpu()" To test on a new dataset, replace generated_test_csv or provided_test_csv with your desired file path.
