#!/usr/bin/env python
# coding: utf-8

# # FUNCTIONS AND CLASSES FOR AL FINAL PROJECT

# In[1]:


import pydicom as dicom
import os
import cv2
#import PIL # optional
import sys, os, pydicom, shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import pandas as pd
from itertools import combinations 
import tensorflow as tf
#tf.enable_eager_execution()
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Input,concatenate, multiply, Reshape, Lambda, Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout, BatchNormalization, AveragePooling2D,ZeroPadding2D

from keras import regularizers, optimizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KernelDensity


# # Functions

# In[2]:


# make it True if you want in PNG format
def convert_dicom(folder_path_from, folder_path_to, PNG = False):
    """ Converts dicom image format to jpg or png
        folder_path_from -- path to get images from for conversion
        folder_path_to   -- path to copy imgs to
    """
    images_path = os.listdir(folder_path_from)
    for n, image in enumerate(images_path):
        ds = dicom.dcmread(os.path.join(folder_path_from, image))
        pixel_array_numpy = ds.pixel_array
        if PNG == False:
            image = image.replace('.dcm', '.jpg')
        else:
            image = image.replace('.dcm', '.png')
        cv2.imwrite(os.path.join(folder_path_to, image), pixel_array_numpy)
        #os.remove(os.path.join(folder_path_from, image))
        
    return None


def remove_damage_img(df1, path):
    """remove damage image(s) that couldn't be read by pydicom or images that
       are not 512, 512 from dataframe.
       returns: tuple of  (number of images remove, modified df)
       Warning: run function only once. 
    """
    df = df1.copy()
    dcm_img_path = []     # list to store path to images
    w = 512
    l = 512
    
    num_of_damage_img = 0
    
    for file in os.listdir(path):
        if file.endswith(".dcm"):
            dcm_img_path.append(file)
            
    dcm_img_path = [os.path.join(path, file) for file in dcm_img_path]
    
    
    for image in dcm_img_path:
        # delete images the program cannot read
        try:
    
            im = pydicom.read_file(image).pixel_array
        
        except: 
            
            #os.remove(image)
            df.drop(df[df['ID'] == image.split('//')[-1]].index.tolist() , inplace = True)   
    
        if im.shape != (w, l):
        
            df.drop(df[df['ID'] == image.split('//')[-1]].index.tolist() , inplace = True)
        
        
    return df


def randomly_select_img(df, size_per_cls):
    """randomly select image index from image dataframe and return a list
       of indices to slice dataframe. It selects images to balance the classes
       """
    
    # list to hold the randomly generated index
    index_list = []
    c = {'epidural' , 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'}

    # the number of combinations
    for i in range(1, 6):
        
        comb = combinations(df.columns[2:], i)
        
        for j in comb: # run combination
        
            hold_format = []
            no_disease = c.difference(set(j))
            
            for no_dis in no_disease:
                hold_format.append([no_dis, '==', 0])
                
            for dis in set(j):
                hold_format.append([dis, '!=', 0])
                
                
            sql = """({} {} {}) & ({} {} {}) & ({} {} {}) & ({} {} {}) & ({} {} {})
                  """.format(*itertools.chain.from_iterable(hold_format))
    
            
            try:   
                result = np.random.choice(df.query(sql).index.tolist(), size = size_per_cls, replace = False)
                
            # if len(df.filter) < size_per_cls return the whole index
            except ValueError: 
                
                result = df.query(sql).index.tolist()
                
            # add indices to kindex list
            for res in result:
                index_list.append(res)
            
            
        #continue
                
    # add index of with no disease
    result = np.random.choice(df.query('any == 0').index.tolist(), size = len(index_list), replace = False)
        
    for res in result:
        index_list.append(res)
                
        
    return index_list


def copy_img_to_folder(df, path_sou, path_des):
    """copy images(filename) in dataframe to a destinated folder denoted 
       by path_des. Assumes that the column to extract the filenames from
       is 'ID'
    """
    
    for file in df["ID"]:
        shutil.copyfile(os.path.join(path, file), os.path.join(path_des, file))
    
    return None


def train_test_split_image(mainPath, test_size):
    """Creates subfolders for training and test split for image data set
       by randomly drawing without replacement from each class/label.
    
       Input: mainPath -- path to the main/parent folder of the image data
              test_size -- percentage of image to split for test
              
       Return:  paths to the parent folders of the image test data and image train data
                These are the paths you give to image generator (keras)
       
       Warning: This function creates a copy of the image data so watch out if you have lots of images.
                You might also want to delete the parent folders created by this function after you're done with 
                this assignment (since it creates a duplicate of images you already have).
                
       This is how you call the function:
            X_train_path, X_test_path = train_test_split_image("your_path", test_size = 0.2)
    """
    import os
    import shutil
    
    # number of folders in mainPath
    num_classes = len(os.listdir(mainPath))
    
    # Labels or class or taget list
    labels      = os.listdir(mainPath)
    
    # main folder names
    folder_name_train = mainPath.split('\\')[-1:][0] + '_train'
    folder_name_test  = mainPath.split('\\')[-1:][0] + '_test'
    
    # main folder path
    train_path = mainPath.split('\\')[:-1]
    train_path.append(folder_name_train)
    test_path  = mainPath.split('\\')[:-1]
    test_path.append(folder_name_test)
    # create path name
    train_path = '\\'.join(train_path)
    test_path  = '\\'.join(test_path)
    
    # create main folders for train and test data
    for path in [train_path, test_path]:
        success = False
        if not os.path.exists(path):
            os.mkdir(path)
            success = True
        else:    
            print("Directory " , path ,  " already exists")
            
    # create subfolders under the folders just created
    if success:
        for class_label in labels:
            os.mkdir(train_path +'\\'+ class_label)
            os.mkdir(test_path  +'\\'+ class_label)
            
    # A dict to hold the number of images to extract for test size -- key:class, value: test_size
    num_test_images = {}
    
    for class_name in labels:
        images  = os.listdir(mainPath +'\\'+ class_name)
        count = len(images)
        num_test_images[class_name] = int(count * test_size)
        
        # randomly pick images for test without replacement
        test_set = np.random.choice(images, size = num_test_images[class_name], replace = False)
        
        # copy images to train and test folder
        # copy to test folder
        for i in test_set:
            shutil.copy(src = mainPath +'\\'+ class_name + '\\' + i , dst = test_path +'\\' + class_name + '\\' + i )
        
        # copy remaining images to training folder
        for j in set(images).difference(set(test_set)):
            shutil.copy(src = mainPath +'\\'+ class_name + '\\' + j , dst = train_path +'\\'+ class_name + '\\' + j )
            
            
    return train_path, test_path  



def onehot(df, columns = ['no_disease', 'disease']):
    """one hot encodes a column vector of 1s and 0s
       returns None type. Works on the dataframe as not object
    """
    encode   = OneHotEncoder(categories='auto')
    y_onehot = encode.fit_transform(df.loc[:, 'any'].values.reshape(-1,1))
    y_onehot = pd.DataFrame(y_onehot.toarray()).astype(int)
    y_onehot.columns = columns
    
    for i in columns:
        df[i] = y_onehot[i]
        
    return None


def graph_wight_violin(dict_wght, fig_size=(15,11)):
    """Graph two violin plots: one for the weights and another for bias
       df_weight -- dictionary of weights and biases
       Returns: plots
    """
    sns.set(rc={'figure.figsize':fig_size})
    sns.set_context("poster")
    
    dict_wght_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_wght.items()]))

    sns.set()#style="whitegrid")
    
    ax = sns.violinplot(x="Groups", y="Values", data=dict_wght_df.melt(var_name='Groups', value_name='Values'), inner='box')
                         
    ax.set_title('Distribution of Weights and Biases')
    
    return plt.show()


# In[3]:


#path_folder = r"D:\ai_final_train_img_2"
#path_folder_to = r"D:\ai_final_train_img_2_jpg"
#df_path   = r"D:\df_2"

#convert_dicom(path_folder, path_folder_to)

# you are only using 2 columns
#df = pd.read_excel(df_path, usecols = ['ID', 'any'], dtypes = {'ID':str, 'any':int})


# # Classes

# In[4]:


class image_gen_ai: #( tf.keras.preprocessing.image):
    
    def __init__(self, image_size, batch, class_mod):
        self.target_size = image_size
        self.batch_size  = batch
        self.class_mode  = class_mod
    
    
    def create_train_gen(self, df, path, y_col = ['no_disease', 'disease']):
        
        train        = ImageDataGenerator(rescale           = 1./255, 
                                          shear_range       = 0.2, 
                                          zoom_range        = 0.2,
                                          rotation_range    = 15,
                                          width_shift_range = 0.2,
                                          height_shift_range= 0.2,
                                          horizontal_flip   = True
                                         )
        
        train_gen   = train.flow_from_dataframe(
                                                dataframe  = df,
                                                directory  = path,
                                                x_col      = "ID",
                                                y_col      = y_col,
                                                batch_size = self.batch_size,
                                                seed       = 42,
                                                shuffle    = True,
                                                class_mode = self.class_mode,
                                                target_size= self.target_size,
                                                validate_filenames=False
                                               )
            
            
        return train_gen
    
    
    def train_gen_model2(self, df, path, y_col = ['epidural','intraparenchymal', 'intraventricular','subarachnoid','subdural']):
        
        train        = ImageDataGenerator(rescale           = 1./255, 
                                          shear_range       = 0.2, 
                                          zoom_range        = 0.2,
                                          rotation_range    = 15,
                                          width_shift_range = 0.2,
                                          height_shift_range= 0.2,
                                          horizontal_flip   = True
                                         )
        
        train_gen2   = train.flow_from_dataframe(
                                                dataframe  = df,
                                                directory  = path,
                                                x_col      = "ID",
                                                y_col      = y_col,
                                                batch_size = self.batch_size,
                                                seed       = 42,
                                                shuffle    = False,
                                                class_mode = self.class_mode,
                                                target_size= self.target_size,
                                                validate_filenames=False
                                               )
        return train_gen2
            
    
    
    def create_val_gen(self, df, path, y_col = ['no_disease', 'disease']):
        
        val        = ImageDataGenerator(rescale           = 1./255, 
                                          shear_range       = 0.2, 
                                          zoom_range        = 0.2,
                                          rotation_range    = 15,
                                          width_shift_range = 0.2,
                                          height_shift_range= 0.2,
                                          horizontal_flip   = True
                                        )
        
        val_gen   = train.flow_from_dataframe(
                                                dataframe  = df,
                                                directory  = path,
                                                x_col      = "ID",
                                                y_col      = y_col,
                                                batch_size = self.batch_size,
                                                shuffle    = True,
                                                class_mode = self.class_mode,
                                                target_size= self.target_size,
                                                validate_filenames=False
                                               )
            
            
        return val_gen
    
    
    def create_test_gen(self, df, path):
        
        test           = ImageDataGenerator(rescale=1./255)
        
        test_gen       = test.flow_from_dataframe(
                                                  dataframe   = df,
                                                  directory   = path,
                                                  x_col       = "ID",
                                                  y_col       = None,
                                                  batch_size  = self.batch_size,
                                                  shuffle     = False,
                                                  class_mode  = None,
                                                  target_size = self.target_size,
                                                  validate_filenames=False
                                                 )
        return test_gen
    
    
    def create_mid_gen(self, df, path):
        """Creates a generator object for extracting middle layers
        """
        
        train        = ImageDataGenerator(rescale = 1./255)
        
        train_gen   = train.flow_from_dataframe(
                                                dataframe  = df,
                                                directory  = path,
                                                x_col      = "ID",
                                                y_col      = None,
                                                batch_size = self.batch_size,
                                                shuffle    = False,
                                                class_mode = None,
                                                target_size= self.target_size,
                                                validate_filenames=False
                                               )
            
            
        return train_gen
       


# In[5]:


class extract_train_replace:
    
    def __init__(self, old_model, begining_layer, ending_layer):
        self.begining_layer    = begining_layer
        self.ending_layer      = ending_layer
        self.old_model         = old_model
        self.old_model_layer   = {v.name: i for i, v in enumerate(old_model.layers)}
        self.out_shape         = model.layers[self.old_model_layer[ending_layer]].output_shape[1]
    

    def get_z_code(self, generator):
        
        """Get z code from 2 different layers.
           Input model     --> model to extract z code from
                 generator --> generator object to iterate over the imgs
                 in_layer  --> layer name as string -- to get the first z code
                 out_layer --> layer name as string -- to get the last z code
           Output two arrays of data --> first z code, second z code
        """
        
        # Extract first z-code
        m1_mid_model_input   = Model(inputs = self.old_model.input, 
                                     outputs = self.old_model.get_layer(self.begining_layer).output
                                    )
        
        # images generator
        generator.reset()
        x_train              = m1_mid_model_input.predict_generator(generator)
        
        
        # Extract second z-code
        m1_mid_model_output = Sequential(name = "trash")
        
        # counter for getting layer index
        count = -1
        
        for i in self.old_model.layers[ : self.old_model_layer[self.ending_layer] + 1]:
            # add layers to new model using old model layers
            m1_mid_model_output.add(i)
            count +=1
            # set weights to old model's weights
            m1_mid_model_output.layers[count].set_weights(i.get_weights())
          
        # fatten prediction to make loss easy
        m1_mid_model_output.add(Flatten(name = 'flatten_output'))
        
        
        # images generator
        generator.reset()
        y_train              = m1_mid_model_output.predict_generator(generator)
        
        del m1_mid_model_output, count, m1_mid_model_input
        
        return x_train, y_train
    
        
    def find_output_dim(self, inputs, filte_r):
        
        """Calculate output of a convoluted network with constraint
           Returns : None if no padding and striding was found or (stride, padding)
           If multiple paddings and stridings where found return the tuple with the minimum
           strides
        """
        lista = []
        minimum = 5
        index_of_min = None
        # go over possible stride value
        for i in range(1, 4):
            # go over possibe padding number
            for j in range(4):
                if (inputs - filte_r + 2*j)/i + 1 == self.out_shape:
                    lista.append((i, j))
                    
        assert len(lista) != 0
            
        if len(lista) == 1: 
            stride, pad = lista[0] 
        
        # get the combination with minimum strides
        else:
            for element in range(len(lista)):
                if minimum >= lista[element][0]:
                    minimum = lista[element][0]
                    index_of_min = element
                    
            stride, pad = lista[index_of_min]
            
            
        # parse the padding and strides so keras can understand    
        
        if  pad  == 1: pad = ((1,0),(0,1))
        elif pad == 2: pad = ((1,1),(1,1))
        elif pad == 3: pad = ((1,1),(2,2))
        elif pad == 4: pad = ((2,2),(2,2))
            
        if  stride  == 1: stride = (1,1)
        elif stride == 2: stride = (2,2)
        elif stride == 3: stride = (3,3)
        elif stride == 4: stride = (4,4)
        
        
        return stride, pad
    
    
    def replace_middle_layer(self, replacement_model):
        """Replaces the middle layer of a model with a completely new model
           
           from_model_layer -- new model serving as the replacement
           to_model_layer   -- model middle layer needing replacement
        """
        replc_model_layer_dict = {v.name: i for i, v in enumerate(replacement_model.layers)}
        
        index_layer = -1
        
         # old model recontruction
        old_model_recontructed = Sequential(name = 'replc_model')
        
        for old_layer in self.old_model.layers:
            index_layer +=1
            
            # if the old model layer come before the beginning-layer I extracted the z-code
            # from, copy the layers and weights to the turucated/reconcontructed model
            if index_layer <= int(self.old_model_layer[self.begining_layer]):
                old_model_recontructed.add(old_layer)
                old_model_recontructed.layers[index_layer].set_weights(old_layer.get_weights())
                
            else: break
                
        # if this is a middle-layer I extracted the z-code from, copy the layers and 
        # weights from the replacement-model to the truncated/recontructed model 
        # add layers to the truncated model using the replacement model excluding the flattened layer.
        for replc_layer in replacement_model.layers[:-1]:
            old_model_recontructed.add(replc_layer)
                    # set old model weight to new model weight
            old_model_recontructed.layers[index_layer].set_weights(replc_layer.get_weights())
            
            index_layer += 1
            
            
        
        # set the remaining weights after the ending layers
        for l in self.old_model.layers[self.old_model_layer[self.ending_layer] + 1 :]:
                
            old_model_recontructed.add(l)
                
            old_model_recontructed.layers[index_layer].set_weights(l.get_weights())
            index_layer +=1
            
            
        return old_model_recontructed
    
    
    def get_weight_bias_df(self):
        """Get weights and biases from old model layer
           returns a dictionary with layer name + weights/biases + an extract index
           just in case the layer doesn't have a name
        """
        
        dict_wght  = {}
        
        for index , lay in enumerate(self.old_model.layers[self.old_model_layer[self.begining_layer]                                       :self.old_model_layer[self.ending_layer] + 1]):
            try:
                dict_wght[lay.name + '_w{}'.format(index)] = lay.get_weights()[0]
                dict_wght[lay.name + '_b{}'.format(index)] = lay.get_weights()[1]
            except IndexError:
                continue
         
        # unravels the nested list of the weights and biases
        for key in dict_wght.keys():
            dict_wght[key] = dict_wght[key].ravel()
    
        
        return dict_wght
    
    
    def z_code_mod_final(self, model, generator, last_layer, df):
        """Get z code from the first model and write it to dataframe.
           Input model      --> model to extract z code from
                 generator  --> generator object to iterate over the imgs with any==1, shuffle must be false
                 last_layer --> layer name as string -- to get z-code from
                 df         --> same dataframe pass to the "generator"
           appends the z-code to the dataframe and writes to the current directory
        """
        # Extract probability of having the disease
        generator.reset()
        x_proba   = model.predict_generator(generator)
        
        # Extract second z-code
        model_layer   = {v.name: i for i, v in enumerate(model.layers)}
        model_temp = Sequential(name = "trash2")
        
        # counter for getting layer index
        count = -1
        
        for i in model.layers[ : model_layer[last_layer] + 1]:
            # add layers to new model using old model layers
            model_temp.add(i)
            count +=1
            # set weights to old model's weights
            model_temp.layers[count].set_weights(i.get_weights())
          
      
        # images generator
        generator.reset()
        x_train              = model_temp.predict_generator(generator)
        
        
        # very important -- multiply all data by their respective predicted probabilities of having the disease
        x_train   = x_train*x_proba[:, 1].reshape(-1,1)
    
        df_new = pd.DataFrame(x_train)
        
        df_new['epidural']         = df['epidural'].values
        df_new['intraparenchymal'] = df['intraparenchymal'].values
        df_new['intraventricular'] = df['intraventricular'].values 
        df_new['subarachnoid']     = df['subarachnoid'].values 
        df_new['subdural']         = df['subdural'].values
        
        p = "train_data_model2.csv"
        # write to the current directory
        df_new.to_csv(p, index = False)
        
        del model_temp, df_new, x_train, x_proba
        
        return p
        
        
        


# In[6]:


class kde_estimate:
    
    def __init__(self):
        pass
        # stores the random 
        #for keys in data_dict.keys:
        #self.out_dict[keys]  = []
        
        
    def my_init_wgt(self, shape, dtype=None):
        """Get samples from the fitted kernel and use as initializer"""
        
        return self.find_kernel(weights=True).sample(shape) #K.random_normal(shape, dtype=dtype)

        
    def my_init_bias(self, shape, dtype=None):
        """Get samples from the fitted kernel and use as initializer"""
        
        return self.find_kernel().sample(shape)
    
    
    def find_kernel(self, data, weights= False):
        """find kernel distribution for the weights and biases
        """
        self.data = data
        if not weights:
            k_bais = KernelDensity(bandwidth=0.7, kernel='gaussian', metric= 'euclidean').fit(data)
            return k_bais
        else:
            k_wght = KernelDensity(bandwidth=0.7, kernel='gaussian', metric= 'euclidean').fit(data)
            return k_wght
        
        
        
    


# In[7]:


#kde = kde_estimate()


# In[ ]:


# RUN TO MAKE TRAINING IMGs
#df = pd.read_csv('ai_final_project_img_df.csv')
#index = randomly_select_img(df, 20)
#path = r'C:\Users\kwewe\Downloads\rsna-intracranial-hemorrhage-detection\stage_1_train_images'
#copy_img_to_folder(df.iloc[index, :], path, r"C:\Users\kwewe\Documents\DATA\ai_final_train_img_2")
#df.iloc[index, :].to_csv(r"C:\Users\kwewe\Documents\DATA\ai_final_train_img_2\df.csv", index = False)

#len(index)
#val_index = np.random.choice(index, 114)
#for i in index:
#    if i in val_index:
#        index.remove(i)
#        
#test_index  = np.random.choice(index, 50)
#for i in index:
#    if i in test_index:
#        index.remove(i)
#
## write the dataframes to a single path
#df.iloc[val_index, :].to_csv(r"D:\final_AI_project_data\val_df.csv", index = False)
#df.iloc[test_index, :].to_csv(r"D:\final_AI_project_data\test_df.csv", index = False)
#df.iloc[index, :].to_csv(r"D:\final_AI_project_data\train_df.csv", index = False)
#

## copy imgs to their repective folders
#copy_img_to_folder(df.iloc[val_index, :], path, r"D:\final_AI_project_data\imgs\val")
#copy_img_to_folder(df.iloc[test_index, :], path, r"D:\final_AI_project_data\imgs\test")
#copy_img_to_folder(df.iloc[index, :], path, r"D:\final_AI_project_data\imgs\train")


# In[ ]:




