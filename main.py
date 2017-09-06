#### Filippo Pedrazzini, David Tomic
#### 5G and Deep Learning


"""
TODO Filippo (bulshit initial things, that I will explain you later.)

1. Importing the dataset -- In the beginning in order to speedup the computation
2. Data visualization with tableau (I ll show you the sw next time)

"""


"""
TODO David

1. download Anaconda Mac.
2. create a python environment (version 3.5) (google it)
3. activate the environment (using the terminal) and install these libraries:
    
    scikitlearn
    tensorflow
    pandas
    keras
    jupyter notebook

4. download pycharm ce and clone the repo 
5. change the interpreter of the project to the environment that you created before.

"""

###### importing the libraries

import pandas as pd


###### importing the dataset


col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"]

dataset = pd.read_csv('./dataset_kdd/train_20.csv', delimiter=',',header=None, names=col_names, index_col=False)

print(dataset.head())
print(dataset.describe())


### Creating the excel file in order to visualize the data in tableau
dataset.to_excel('train.xlsx')




###### train - test split




###### create the model





###### validate the model





###### test the model



