#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.applications import VGG16
import keras as k


# In[2]:


# get imagenet model
vgg_conv = VGG16(weights='imagenet',  include_top=False, input_shape=(224, 224, 3) )
vgg_conv.summary()


# In[3]:


# remove the last 2 layers
vgg_conv.layers.pop()
vgg_conv.layers.pop() 
vgg_conv.summary()


# In[17]:


# build your own transfer learning model
last= vgg_conv.get_layer('block5_conv2').output
x= k.layers.Flatten()(last)
x2= k.layers.Dense(1024, activation='relu')(x)   # Fully Connect
my_preds= k.layers.Dense(200, activation='softmax')(x2)# combinemodified VGG with your FC+Softmax
my_model= k.Model(vgg_conv.input, my_preds)


# make all other layers non-trainable
for layer in my_model.layers[:10]:#    
    layer.trainable= False      
#    Dense(32, trainable=False)
my_model.summary()

# Run and train model
my_model.compile(loss='crossentropy', optimizer='adam') 

