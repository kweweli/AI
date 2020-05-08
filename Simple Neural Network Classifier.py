#!/usr/bin/env python
# coding: utf-8

# ## Dependencies

# In[3]:


import pandas as pd
import numpy as np
import tensorflow as tf
import keras as k
from sklearn.preprocessing import StandardScaler as standardized
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score   
from sklearn.metrics import roc_curve


# ## Functions

# In[4]:


def graphLearningCurve(title, loss, cv_loss, epoch, ylim=None):
    """Generate a simple plot of the validation and training learning curve"""
   
    plt.figure(figsize = (8, 6))
    plt.title(title)
    plt.xlim(0, 1+epoch)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.grid()
    epochs = [i for i in range(1, epoch+1)]
    epochs.reverse()
    plt.plot(epochs, loss,    'o-', color="r",   label="Training Loss")
    plt.plot(epochs, cv_loss, 'o-', color="g", label="Cross-validation Loss")

    plt.legend(loc="best")
    return plt

def plot_roc(title, y_test, pred):
    """Plots Roc cuve
    
       Parameters
        -------------
       Input: title  -- graph title
       Imput: y_test -- array or list of output test set
       Input: pred   -- array or list of predictions
       Return 
       graph
    """
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    
    plt.figure(figsize = (8, 6))
    plt.title(title)
   
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr,        tpr, '-', color="g",  label="ROC Curve")
    plt.plot(thresholds, thresholds, 'k--', label="Threshold")
    
    plt.legend(loc="best")
    
    return plt

def buid_train_model(X,
                     y,
                     epoch      = 60 ,
                     optimizer  = 'adam' ,
                     loss       = 'binary_crossentropy' ,
                     batch_size = 20 ,
                     act        = 'relu' ,
                     last_act   = 'sigmoid'):
    
    """Build and Train a simple NN.

        Parameters (see Keras Doc for explanation of parameters)
        ----------
        X         : array, required -- input data
        y         : array, required -- output data
        epoch     : int, required
        optimizer : string, required
        loss      : string, required
        loss      : int, required
        act       : string, required -- activation to use
        act_last  : string, required -- activation for output layer
        
        Returns
        Model object
    """

    # Build Model
    model = k.Sequential()
    model.add(k.layers.Dense(13, input_dim = 13, activation =  act))
    model.add(k.layers.Dense(8, activation = act))
    model.add(k.layers.Dense(8, activation = act))
    model.add(k.layers.Dense(6, activation = act))
    model.add(k.layers.Dense(1, activation = last_act))
    # compile model
    model.compile(optimizer = optimizer,
                  loss     = loss)
    # train model
    model.fit(X, 
              y,   
              batch_size       = batch_size , 
              epochs           = epoch, 
              validation_split = 0.1, 
              class_weight     = {1:2, 0:1}, 
              verbose          = 0)
    
    return model


# ## Data Preparation
# Read data, slice predictors and label, and transform label to 1's and 0's
# Do a train_test_split of 10%

# In[5]:


# read file and segment target and predicts
data = pd.read_csv('cellDNA.csv', header = None)

X = data.iloc[:,:-1]
y = np.array([1 if i != 0 else 0 for i in data.iloc[:, -1:].values]).reshape(data.shape[0], -1)

# train, test, split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 0)

# normalized X_train, and X_test
normalized            = standardized()
X_train_norm          = normalized.fit_transform(X_train)
X_test_norm           = normalized.transform(X_test)


# ## Compute common statistics and class sizes
# Class is highly imbalance

# In[6]:


print('Interesting:     ', len(y[y==1]))            # Count of bacteria that are worth studying
print('Not Interesting: ', len(y[y==0]))

display(X.describe())                               # basic stats of predictors


# ## Build Model and Evaluate Learning Curve.
# Added 'class_weight' parameter to the model to account for high class imbalance. 
# The interesting class (i.e., the bacterium worth studying or 1) is very small as compared to the other class so it is given a weight of 2. In other words, one data point from class 1 is worth twice from class 0. I tried different values for epochs and 35 seems to be the best in tracking the validation loss (at least for the current model structure). A natural extension might be to try different model structure.

# In[12]:


epoch = 15

# build and train network
model = buid_train_model(X_train_norm, y_train, epoch = epoch )

# Leaning Curve
cv_loss = model.history.history['val_loss'] # validation loss
loss    = model.history.history['loss']     # Training Loss
cv_loss.reverse()                           
loss.reverse()
graphLearningCurve("NN Learning Curve", loss, cv_loss, epoch, ylim=None)       #graph learning curve


# # Loss on Test Set

# In[13]:


print('Loss on Test Set: ', model.evaluate(X_test_norm, y_test))
pred  = model.predict_classes(X_test_norm).ravel()               # predict on test set


# # Metrics
# The model has an accuracy 94%, precision of 73% and Recall of 94%. 

# In[14]:


# dict of metrics
metrics_dict = {'recall': recall_score(y_test, pred), 'precision': precision_score(y_test, pred),
               'accuracy': accuracy_score(y_test, pred)}

# display metrics
print('Recall:    ', metrics_dict['recall'])
print('Precision: ', metrics_dict['precision'])
print('Accuracy:  ', metrics_dict['accuracy'])


# In[15]:


# Plot roc cure
plot_roc("Feedforward NN Roc Curve", y_test, pred )

