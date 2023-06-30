# Credit Card Fraud Detection Using Federated Learning
Download the dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
This dataset has been processed, split into train and test, and it was used for training, testing, and comparison. Smote1.ipynb is to generate somte file for the dataset.
# Required Files:
[creditcard_train_SMOTE_1.csv](https://drive.google.com/file/d/1vEFjrA5I08dVPEslMVixJG0Kjl9bW44V/view)  

[creditcard_test.csv](https://drive.google.com/file/d/1rJlgEOpakousK-83fKKNg9xrjfPvU8sf/view)  

[creditcard_train.csv](https://drive.google.com/file/d/1nnRE2v7J-zt5xyR9dy9QCwE1cShkDKH9/view)  


# How to run it
Step 1: Download the dataset from the above links.

Step 2: Open and run project_smote.py file and once it is executed run project.py file.

Step 3: In the project.py file, change the value of "num_clients" variable, which decides the dataset split size. It should be the number that can divide the dataset length. (ex. 32,24,16,12)

Step 4: Change the variable "num_selected", which decides the number of bank clients. It should be less than "num_clients" variable. 

Step 5: Add dataset to the drive, and replace the path for training and testing file(Variable "traindata" and "test_file"). This should be according to your drive path. Train file for balanced dataset should be of "creditcard_train_SMOTE_1.csv", and for imbalanced dataset it should be of "creditcard_train.csv". For test data, the path should be of "creditcard_test.csv".

Step 6: Run all the cells of project.py file. It will store the results in "mp" variable.

Step 7: Now, run the whole project.py file again by changing number of clients by changing variable "num_selected". This should be done with the same dataset split.

We will get the confusion matrix and graphs the would depict the accuracy, precison, recall and f1 score of our model.
