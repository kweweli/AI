#!/usr/bin/env python
# coding: utf-8

#                                                         AI Homework 5
#                                            Extracting Hidden Representation From NN
# 

# ## Dependencies

# In[1]:


import pandas                as pd
import numpy                 as np
import tensorflow            as tf
import keras                 as k
import matplotlib.pyplot     as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, MaxPooling2D, Conv2D, BatchNormalization, Activation


# ## Functions

# In[2]:


def graphLearningCurve(title, loss, cv_loss, epoch, ylim=None):
    """Generate a simple plot of the validation and training learning curve"""
   
    plt.figure(figsize = (8, 6))
    plt.title(title)
    plt.xlim(0, 1 + epoch)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.grid()
    epochs = [i for i in range(1, epoch+1)]
    epochs.reverse()
    plt.plot(epochs, loss,    'o-', color="r",   label="Training Loss")
    plt.plot(epochs, cv_loss, 'o-', color="g",   label="Cross-validation Loss")

    plt.legend(loc="best")
    return plt


# In[3]:


# train path
X_train_path, X_test_path = r"C:\Users\kwewe\OneDrive - University of St. Thomas\AI\Iris_Imgs_train",                                 r"C:\Users\kwewe\OneDrive - University of St. Thomas\AI\Iris_Imgs_test"


# In[4]:


# https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keras_flow_from_directory.ipynb
# https://github.com/philipperemy/keract


# ## Data Preparation
# Create generator to feed image to model using batch size. I change the epochs to 10 to allow my model to run after so I can submit a result but the optimal epochs is about 160.

# In[5]:


batch_size = 5
epochs     = 10
seed       = 0
# train generator instantiation
train        = ImageDataGenerator(rescale=1./255, 
                                  shear_range=0.2, 
                                  zoom_range=0.2,
                                  featurewise_center=True,
                                  featurewise_std_normalization=True,
                                  rotation_range=15,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  horizontal_flip=True)
# train generator instantiation     
test        = ImageDataGenerator(rescale=1./255)
     
train_generator = train.flow_from_directory(X_train_path, target_size=(200, 200), 
                                            batch_size=batch_size,
                                            class_mode='categorical', 
                                            classes = ['setosa', 'versicolor', 'virginica'])

test_generator = test.flow_from_directory(X_test_path, target_size=(200, 200), 
                                          batch_size= 2,
                                          class_mode='categorical',
                                          classes = ['setosa', 'versicolor', 'virginica']
                                          )


# # Q2 Build Model (with Train-Test Split) 

# In[6]:


# inceptionv3 model instantiation with fc removed
model_v = InceptionV3(include_top = False, weights='imagenet', input_shape=(200, 200, 3), pooling='max')

# freeze all layers up to 'activation_23', inclusive
dictionary = {v.name: i for i, v in enumerate(model_v.layers)}   # dictionaly of index and layers names

for i in model_v.layers[:dictionary['activation_23'] + 1]:
    i.trainable = False
    
# customize model
x             = model_v.get_layer('activation_23').output
x             = Conv2D(32, (3,3),  name = 'my_conv_1')(x)
x             = BatchNormalization(name = 'my_batch_1')(x)
x             = Activation('relu', name = 'my_act_1' )(x)
x             = MaxPooling2D(2,2,  name = 'my_max_pool_1')(x)
x             = Flatten(name = 'my_flatten_1')(x)
x             = Dense(8, name = 'my_fc_1')(x)
x             = BatchNormalization(name = 'my_batch_2')(x)
x             = Activation('relu', name = 'my_act_2')(x)
x             = Dense(5, activation='relu', name = 'my_fc_2')(x)
softmax_layer = Dense(3, activation='softmax', name = 'my_output')(x)

# create concatenated model
model   = k.models.Model(input = model_v.input, output = softmax_layer)
# optimizer
adamN = k.optimizers.Nadam(lr=0.003, beta_1=0.9, beta_2=0.999)

# compile
model.compile(optimizer= adamN, loss='categorical_crossentropy', metrics=['accuracy'])


# number of samples
total_train_samp = train_generator.n
total_val_samp = test_generator.n

# early stoping
early = k.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto',
                                           restore_best_weights = True)


# fit model
model.fit_generator(train_generator, 
                    steps_per_epoch = int(total_train_samp/batch_size), 
                    epochs=epochs, 
                    validation_data = test_generator,
                    validation_steps= int(total_val_samp/2),
                    verbose = 0,
                    callbacks = [early]
                    )


# # Q3 Model Summary

# In[8]:


#print(np.all(model_Zcode.layers[i].get_weights()[0] == adjusted_layer[i].get_weights()[0]))
model.summary()


# # Learning Curve

# In[9]:


# Leaning Curve
cv_loss = model.history.history['val_loss'] # validation loss
loss    =  model.history.history['loss']    # Training Loss
cv_loss.reverse()                           
loss.reverse()
graphLearningCurve("NN Learning Curve", loss, cv_loss, epochs, ylim=None)       #graph learning curve


# # Q4 Confusion Matrix on Test Set

# In[10]:


# Confution Matrix 
Y_pred   = model.predict_generator(test_generator)
y_pred   = np.argmax(Y_pred, axis=1)
conf_mat = confusion_matrix(test_generator.classes, y_pred)
print('Confusion Matrix')
print(conf_mat, '\n')

# Calculate Accuracy on Test data
print('Accuracy: ', str(round((sum(np.diagonal(conf_mat))/total_val_samp)*100, 2))+'%', '\n')

# Classification Report
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, 
                            target_names=['setosa', 'versicolor', 'virginica']))


# # Q5 Prediction on Test Set Plot

# In[11]:


from mpl_toolkits.mplot3d import Axes3D
class_size = 3
label_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

colors = np.array(['#009AFF', '#FF00AB', '#2E8B57'])

fig = plt.figure(figsize=(9,9))
ax1 = fig.add_subplot(111, projection='3d')


for cl in range(class_size):
    indices = np.where(y_pred == cl)
    ax1.scatter(Y_pred[indices,0], Y_pred[indices, 1],Y_pred[indices, 2] , c = colors[cl], 
                label= label_dict[cl])
ax1.legend()
ax1.title.set_text("Classification of Flower Type")
plt.show()

