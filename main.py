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

## the first thing to do is always to import all the libraries that we will use in the code

import pandas as pd
from collections import Counter
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import preprocessing
from keras.utils import np_utils




"""

There are many ways to load the dataset into our project, but the most used when you deal with CSV or TXT is pandas.
It is really fast in the importing phase and has some functions in order to modify a bit the dataset based on your needs.

With these kind of libraries you can also upload images, text, audio and whatever you want.
Here an example in the case you will need it.

from PIL import Image

img = Image.open('path_to_the_image.format')

TRASFORMING THE IMAGE IN ARRAY
imagarray = np.asarray(img)

SHAPE OF THE ARRAY
imagarray.shape  --> (40, 40, 3) in case of images you have (height, width, RGB_value)

FROM MATRIX TO ARRAY
imagarray.ravel().shape

For images it is better to use Keras (the deep learning library that we are going to use) functions that are used mainly for this reasons.
Deep learning infact is really good with unstructure data (sound, text, images), but it is also good with structured data as our problem.

"""


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

"""
Some useful functions to visualize the data:

pandas libary

THIRD ROW
df.iloc[3]

FIRST FIVE ROWS OF DURATION COLUMN
df.loc[0:4, 'duration']

MULTIPLE COLUMNS
df[['duration', 'protocol']]

SELECT THE ROWS WITH...
df[df['duration'] > 70] --> CONDITION CAN BE COMBINED

LIST OF TRUE AND FALSE
df['duration'] > 70

QUERY
df.query("Age > 70")

UNIQUE VALUES
df['duration'].unique()

SORTING
df.sort_values('duration', ascending=False)

matplotlib

plt.plot(df)
plt.title('Line Plot')
plt.legend(['col1','col2','col2','col3'])
plt.show()

ANOTHER WAY, USING DIRECTLY PANDAS
KIND: type of plot
df.plot(kind='scatter', title='', )

Visualize the data is really important and this is just one method.
That's why I am used to export the dataset in Excel and then visualize it using tableau.

"""

### Creating the excel file in order to visualize the data in tableau

# TODO
"""
exporting the dataset with just 2 classes, in order to see if there is a linear correlation between the features

"""
# dataset.to_excel('train.xlsx')

### counting the number of instances for each class

#print(dataset['labels'].value_counts())

### Dropping categorical features

num_features = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

#dataset_num_features = dataset[num_features].astype(float)

### the problem is a binary classification problem -- we ahve to identify if is an attack or not. we don't care
### which is the type of the attack

target = dataset['labels'].copy()
target[target!='normal'] = 'attack'

#X = dataset_num_features.iloc[:, :-1].values
#y = dataset.iloc[:, -1:].values


## variables = dataset.columns[dataset.columns != 'labels']

X = dataset[num_features].astype(float).values
#print(X)

le = preprocessing.LabelEncoder()
le.fit(target)
binary_target = le.transform(target)
y = binary_target
print(y)

#print(X)
#print(y)

###### train - test split

# Splitting the dataset into the Training set and Test set is not required in this case

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print('dataset shape {}'.format(Counter(y_train)))

# Get dimensions of input and output
dimof_input = X_train.shape[1]
dimof_output = np.max(y_train) + 1
print('dimof_input: ', dimof_input)
print('dimof_output: ', dimof_output)

# Set y categorical
y_train = np_utils.to_categorical(y_train, dimof_output)
y_test = np_utils.to_categorical(y_test, dimof_output)


#### data preprocessing


#TODO
# scaling the features
# feature selection

###### create the model




####### KERAS MODEL

# Set constants
batch_size = 128
dimof_middle = 100
dropout = 0.2
count_of_epoch = 100
verbose = 1


print('batch_size: ', batch_size)
print('dimof_middle: ', dimof_middle)
print('dropout: ', dropout)
print('countof_epoch: ', count_of_epoch)
print('verbose: ', verbose)

from keras.models import Sequential

model = Sequential()

#### model design

#TODO
# 1. adding autoencoders
# 2. tuning parameters

from keras.models import Sequential
from keras.layers.core import Dense, Dropout


### dimension of input is about the number of features (like in images where the number of features correspond to the number of pixels
model = Sequential()
model.add(Dense(dimof_middle, input_dim=dimof_input, kernel_initializer='uniform', activation='tanh'))
model.add(Dropout(dropout))
model.add(Dense(dimof_middle, kernel_initializer='uniform', activation='tanh'))
model.add(Dropout(dropout))
model.add(Dense(dimof_output, kernel_initializer='uniform', activation='softmax'))

#### training

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=1, verbose=verbose)



###### validate the model

###### test the model

loss_and_metrics = model.evaluate(X_test, y_test, batch_size=batch_size)

classes = model.predict(X_test, batch_size=128)




