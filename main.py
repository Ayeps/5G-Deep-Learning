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

import numpy as np

"""
NUMPY -- is the library behind all these libraries that we are going to use. 
It is really powerful in computation. Instead of using normal for cycles uses special
methods to do the same task in few seconds instead of hours.

A = np.array([[1,2,3],
             [4,5,6]
             [7,8,9]
             [10,11,12]])
             
The shape is (4,3) --> 4 rows, 3 columns

A[0] --> first row
A[2,0] --> cell of the matrix
A[:,0] --> : every row, the first element --> first column

You can also do operations

A * 3

A + A... but, they must be of the same shape

A.dot(A) --> dot product

"""

###### importing the libraries

## the first thing to do is always to import all the libraries that we will use in the code

import pandas as pd
from collections import Counter
from sklearn.cross_validation import train_test_split

from sklearn import preprocessing
from keras.utils import np_utils

import matplotlib.pyplot as plt


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

"""
another way to visualize the data is to use jupyter notebook, from the terminal is difficult to see all this plots

_ = dataset.hist(figsize=(12,10))

import seaborn as sns

sns.pairplot(dataset, hue='Labels') --> in case of binary
sns.heatmap(dataset.corr(), annot=True)


"""


### Creating the excel file in order to visualize the data in tableau

# TODO
"""
exporting the dataset with just 2 classes, in order to see if there is a linear correlation between the features

"""
# dataset.to_excel('train.xlsx')

### counting the number of instances for each class

#print(dataset['labels'].value_counts())

### Dropping categorical features (for now) --> Like this we can not use them.
### We need one hot encoder -- when we modify something of the X_train is necessary to modify also something of X_test

"""

dataset_cat = pd.get_dummies(dataset)

X = dataset_cat.iloc[:, :-1].astype(float).values ### all the columns a part from the last one
y = dataset_cat.iloc[:, -1:].values #### all the columns starting from the last one

"""

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

X = dataset[num_features].astype(float).values

le = preprocessing.LabelEncoder()
le.fit(target)
binary_target = le.transform(target)
y = binary_target
print(y)

#print(X)
#print(y)


#### data preprocessing


#TODO

# feature selection


#### feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

### checking X
print(X[:10])
print(X.shape)


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

###### create the model




####### KERAS MODEL

# Set constants
batch_size = 128
dimof_middle = 100 ## number of units for each hidden layer
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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


### callbacks to improve the model
checkpointer = ModelCheckpoint(filepath='./models/weights.hdf5', verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir='./models/tensorboard/') ### command for the bash: tensorboard --logdir='./models/tensorboard/'


def build_model():

    ## sequential model because we add elements in a sequence
    model = Sequential()

    """
    
    we can also use dropout in the beginning
    Dropout(0.2, input_shape=X_train.shape[1:]
    
    """

    ### dimension of input is about the number of features (like in images where the number of features correspond to the number of pixels
    model.add(Dense(dimof_middle, input_dim=dimof_input, kernel_initializer='uniform', activation='tanh'))  ## input layer

    ### it is also possible to add regularization into a layer --> kernel_regularize='l2'

    model.add(Dropout(dropout))
    model.add(Dense(dimof_middle, kernel_initializer='uniform', activation='tanh'))

    """
    
    DROPOUT - avoid overfitting
    
    randomly killing some nodes at each iteration using a certain probability. It forces the network to see 
    more robust features. The network will not focus on just few features, because it could happen that in the next iteration 
    this feature will not be present.
    
    It is a parameter to tune (30%-60%)
    
    """
    model.add(Dropout(dropout))
    model.add(Dense(dimof_output, kernel_initializer='uniform', activation='softmax'))  ## output layer (in case of just 2 classes we can use sigmoid)
    """
    
    managing dim_of_output
    
    """

    model.summary()  ## we can check our model
    ## (none, 1) as output, because we can predict the output of many elements at the same time

    W, B = model.get_weights()  ## and get the weights of the model

    #### COMPILE the model --> decide which loss and optimizer to use to train the model

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']) ### in case of multiple classes we have to use categorical_crossentropy

    return model


model = build_model()

##### LEARNING CURVE ON THE NUMBER OF SAMPLES


"""
Before starting the tuning of the model and change it, it is better to see how many samples we need
to perform better. After understanding the number of samples, we can start to design our model.

"""

from sklearn.model_selection import learning_curve
initial_weights = model.get_weights()

train_sizes = (len(X_train) * np.linspace(0.1, 0.99, 4)).astype(int) #### 4 different lenghts --> from 10% to 99%

train_scores = []
test_scores = []

for train_size in train_sizes:
    X_train_frac, _, y_train_frac, _ = train_test_split(X_train, y_train, train_size=train_size)
    model.set_weights(initial_weights)

    h = model.fit(X_train_frac, y_train_frac, verbose=1, epochs=10, callbacks=[EarlyStopping(monitor='loss', patience=1)])

    r = model.evaluate(X_train_frac, y_train_frac, verbose=0)
    train_scores.append(r[-1])

    e = model.evaluate(X_test, y_test, verbose=0)
    test_scores.append(e[-1])

    print("Done size: ", train_size)ù


plt.plot(train_sizes, train_scores, 'o-', label="Training Score")
plt.plot(train_sizes, test_scores, 'o-', label="Test Score")
plt.legend(loc='best')
plt.show()


### second method to do cross validation
##model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=1, verbose=verbose, callbacks=[checkpointer, earlystopper, tensorboard])

## needs a build function
# model = KerasClassifier(build_fn=build_model, epochs=5, verbose=1)

#### first method to do cross validation
# cv = KFold(3, shuffle=True)
# scores = cross_val_score(model, X_train, y_train, cv=cv)


"""
BATCH SIZE

instead of passing one point by one (SGD), we pass a batch (16, 32, 64, 128) of points in order to speed up the computation 
and avoid overfitting.

What happens if we change the batch size?

Big --> slow convergence

Small --> fast convergence

"""
"""
BATCH NORMALIZATION

In order to avoid overfitting

"""

from keras.layers import BatchNormalization

def repeated_training(X_train, y_train, X_test, y_test, units=512, activation='sigmoid', optimizer='sgd', do_bn=False, epochs=10, repeats=3):

    histories = []

    for repeat in range(repeats):
        K.clear_session()

        model = Sequential()

        ### first layer
        model.add(Dense(units, input_shape=X_train.shape[1:], kernel_initializer='normal', activation=activation))

        if do_bn:
            model.add(BatchNormalization)

        ### second layer
        model.add(Dense(units, kernel_initializer='normal', activation=activation))

        if do_bn:
            model.add(BatchNormalization)

        ### third layer
        model.add(Dense(units, kernel_initializer='normal', activation=activation))

        if do_bn:
            model.add(BatchNormalization)

        ### output layer
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])

        h = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=0)

        histories.append([h.history['acc'], h.history['val_acc']])
        print(repeat, end=' ')

    histories = np.array(histories)

    mean_acc = histories.mean(axis=0)
    std_acc = histories.std(axis=0)
    print()

    #### 0 --> training ::: 1 --> test
    return mean_acc[0], std_acc[0], mean_acc[1], std_acc[1]


#### Now we can compare between the model trained with batch normalization and the one without


mean_acc, std_acc, mean_acc_val, std_acc_val = repeated_training(X_train, y_train, X_test, y_test, do_bn=False)

mean_acc_bn, std_acc_bn, mean_acc_val_bn, std_acc_val_bn = repeated_training(X_train, y_train, X_test, y_test, do_bn=True)

### and now we plot all the results
### speed up the training and increase also generalization


def plot_mean_std(m, s):
    plt.plot(m)
    plt.fill_between(range(len(m)), m-s, m+s, alpha=0.1)

plot_mean_std(mean_acc, std_acc)
plot_mean_std(mean_acc_val, std_acc_val)
plot_mean_std(mean_acc_bn, std_acc_bn)
plot_mean_std(mean_acc_val_bn, std_acc_val_bn)

plt.ylim(0, 1.01)
plt.title("Batch Normalization Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test', 'Train_bn', 'Test_bn'])

plt.show()



###### validate the model and tune the parameters

"""
we can tune many different things:

    learning rate
    optimizer
    weights initialization
    batch size...

"""

#TODO

# hyperparameter search

# tune the parameters (gridsearch, random, bayesian... for keras)


###### predict

## exactly as in scikitlearn
## model.predict(X_test).ravel() --> to flat the column into array


### prediction, but are probabilities to belong to the class
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)


#### classification report

from sklearn.metrics import classification_report

print(classification_report(y_test_classes, y_pred_classes))

#### Confusion Matrix

def custom_confusion_matrix(y_true, y_pred, labels=["False", "True"]):
    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ["Predicted " + l for l in labels]
    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
    return df


print(custom_confusion_matrix(y_test, y_pred_classes, ['Attack', 'Normal']))

result = model.evaluate(X_test, y_test, batch_size=batch_size)

print(result)

### visualizing the layers of the model
print(model.layers)
inp = model.layers[0].input
out = model.layers[0].output

print(inp) ### tensors
print(out)


#### MULTIPLE CLASS CLASSIFICATION
"""

In the case that we have to separate all the different classes, 
we need to define in output layer the number of classes.

target_names = dataset['labels'].unique()
print(target_names)

target_dict = {n:i for i,n in enumerate(target_name)}
print(target_dict)

y = dataset['labels'].map(target_dict)
print(y.head())

y_cat = to_categorical(y)
print(y_cat[:10]) ## first ten rows


"""


