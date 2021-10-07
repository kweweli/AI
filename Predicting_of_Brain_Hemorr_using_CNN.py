#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'AI_final_project_module.ipynb')
import warnings
warnings.filterwarnings('once')


# In[2]:


batch_size = 1
epochs     = 10
img_size       = (512, 512)

# DATA
train_data       = pd.read_csv(r"D:\final_AI_project_data\train_df.csv") # slice of data for train
val_data         = pd.read_csv(r"D:\final_AI_project_data\val_df.csv") # slice of data for test
test_data        = pd.read_csv(r"D:\final_AI_project_data\test_df.csv")

onehot(train_data)
onehot(test_data)
onehot(val_data)

# GENERATOR
# Instantiate image generator ai class
image_gen_ai     = image_gen_ai(img_size, batch_size, 'raw')

train_gen        = image_gen_ai.create_train_gen(train_data, r"D:\final_AI_project_data\imgs\train")
test_gen         = image_gen_ai.create_test_gen(test_data, r"D:\final_AI_project_data\imgs\test")
val_gen          = image_gen_ai.create_train_gen(val_data, r"D:\final_AI_project_data\imgs\val")


#path = r'C:\Users\kwewe\Downloads\rsna-intracranial-hemorrhage-detection\stage_1_train_images'
#img_dataframe_path = r'C:\Users\kwewe\OneDrive - University of St. Thomas\AI\ai_final_project_img_df.csv'
#im = pydicom.read_file(os.path.join(path, 'ID_0cce38bf8.dcm')).pixel_array


# In[3]:


model = Sequential(name='model')
# convolution 1
model.add(Conv2D(16, (3,3), activation='relu', input_shape =(img_size[0], img_size[1], 3), name = 'conv_1'))
model.add(BatchNormalization(name = 'batch_1'))
model.add(Activation('relu', name = 'act_1' ))
model.add(MaxPooling2D(2,2,  name = 'max_pool_1'))
# The second convolution
model.add(Conv2D(32, (3,3),  name = 'conv_2'))
model.add(BatchNormalization(name = 'batch_2'))
model.add(Activation('relu', name = 'act_2' ))
model.add(MaxPooling2D(2,2,  name = 'max_pool_2'))
# The third convolution
model.add(Conv2D(64, (3,3), name = 'conv_3'))
model.add(BatchNormalization(name = 'batch_3'))
model.add(Activation('relu', name = 'act_3' ))
model.add(MaxPooling2D(2,2, name = 'max_pool_3'))
# The fourth convolution
model.add(Conv2D(64, (3,3), name =  'conv_4'))
model.add(BatchNormalization(name = 'batch_4'))
model.add(Activation('relu', name = 'act_4' ))
model.add(MaxPooling2D(2,2, name = 'max_pool_4'))
# The fifth convolution
model.add(Conv2D(128, (3,3), name =  'conv_5'))
model.add(BatchNormalization(name = 'batch_5'))
model.add(Activation('relu', name = 'act_5' ))
model.add(MaxPooling2D(2,2, name = 'max_pool_5'))
# Flatten the results to feed into a dense layer
model.add(Flatten(name = 'flatten_6'))
# 100 neuron in the fully-connected layer
model.add(Dense(100, name = 'fc_6'))
model.add(BatchNormalization(name = 'batch_6'))
model.add(Activation('relu', name = 'act_6'))
# 50 neuron in the third fully-connected layer
model.add(Dense(50, name = 'fc_7'))
model.add(BatchNormalization(name = 'batch_7'))
model.add(Activation('relu', name = 'act_7'))
# 2 output neurons for 2 classes with the softmax activation
model.add(Dense(2, activation='softmax', name = 'output_disease'))

# optimizer
adamN = optimizers.Nadam(lr=0.004, beta_1=0.9, beta_2=0.999)

# compile
model.compile(optimizer= adamN, loss= 'categorical_crossentropy', metrics=['accuracy'])

checkpoint_path = r"C:\Users\kwewe\OneDrive - University of St. Thomas\AI"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=1)


# fit model
model.fit_generator(train_gen, 
                    steps_per_epoch = train_gen.n//train_gen.batch_size, 
                    epochs          = epochs,
                    validation_data = val_gen,
                    validation_steps= val_gen.n//val_gen.batch_size,
                    verbose         = 1
                    
                    )

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))


# In[4]:


model.summary()


# # Train "model"  extracted z-code on a new model

# In[5]:


mid_gen = image_gen_ai.create_mid_gen(train_data, r"D:\final_AI_project_data\imgs\train")

# Instantiate Extraction class
extract = extract_train_replace(model, begining_layer='act_2', ending_layer='conv_5')

# get z code
m1_mid_model_input, m1_mid_model_output = extract.get_z_code(mid_gen)
    


# In[6]:


# weights and biases dictionary for middle layers
dict_wght = extract.get_weight_bias_df()
# Graph weights of layers
graph_wight_violin(dict_wght, fig_size=(14,8))


# In[7]:


# Buid middle extracted layer model
# make output and inputs to fill in automatically

# find striding and padding to make dimension work
stride, pad = extract.find_output_dim(inputs = 15, filte_r = 3)



model2 = Sequential(name = 'model_1_mid')
# The first convolution with input sizes same as m1_mid_model_input
model2.add(Conv2D(m1_mid_model_input.shape[-1], (3,3), activation='relu', 
                  input_shape = m1_mid_model_input.shape[1:], name = 'mid_conv_1'))
model2.add(Activation('relu', name = 'mid_act_1' ))
model2.add(MaxPooling2D(2, 2, name = 'mid_max_pool_1'))
# The second convolution
model2.add(Conv2D(64, (3,3),  name = 'mid_conv_2'))
model2.add(Activation('relu', name = 'mid_act_2' ))
model2.add(MaxPooling2D(3,3,  name = 'mid_max_pool_2'))
# The third convolution
model2.add(Conv2D(128, (3,3), name = 'mid_conv_3', padding = 'valid', strides = stride))
model2.add(Activation('relu', name = 'mid_act_3' ))
model2.add(ZeroPadding2D(padding = pad, name = 'mid_pad_3'))
#model2.add(MaxPooling2D(2, 2, name = 'max_pool_3'))
# Flatten the results to feed into a dense layer
model2.add(Flatten(name = 'mid_flatten_4'))



epochs_mid = epochs + 10

# compile
model2.compile(optimizer= adamN, loss = tf.keras.losses.MeanAbsoluteError())

model2.fit(m1_mid_model_input, m1_mid_model_output,  batch_size=1, epochs= epochs_mid)

model2.summary()


# # Reconstruct "model" (by replacing the middle extracted layer with the new middle-layer)

# In[8]:


model1_replc = extract.replace_middle_layer(model2)
model1_replc.summary()


# # Building second model: Final Model

# In[9]:


epochs = 20
batch_size = 3
# generator
# only pass data where any=1
df_disease =  train_data[train_data['any']==1].copy()

train_gen_disease   = image_gen_ai.train_gen_model2(df = df_disease, path =  r"D:\final_AI_project_data\imgs\train")

# filename to get data for training final model

train2_path  = extract.z_code_mod_final(model=model1_replc, generator=train_gen_disease, last_layer='fc_7', df= df_disease)

# delete df_disease to clear memory
del df_disease

data  = pd.read_csv(train2_path)

last_model = Sequential(name = 'model_Final')
last_model.add(Dense(25, input_dim=(50), name = 'den_1')) # input = dimension of data extracted from model1_replc
last_model.add(BatchNormalization(name = 'batch_1'))
last_model.add(Activation('relu', name = 'act_1' ))
last_model.add(Dense(10, name = 'den_2'))
last_model.add(BatchNormalization(name = 'batch_2'))
last_model.add(Activation('relu', name = 'act_2' ))
last_model.add(Dense(5, activation = 'sigmoid', name = 'out_final'))

# compile
last_model.compile(optimizer= adamN, loss= 'mse')

# fit model
last_model.fit(data.iloc[:,:-5], data.iloc[:,-5:],  batch_size= batch_size, epochs=epochs, 
          validation_split=0.1, verbose=0)

last_model.summary()


# # Prediction on Model 1

# In[11]:


test_gen.reset()
# Confution Matrix 
Y_pred   = model1_replc.predict_generator(test_gen)
y_pred   = np.argmax(Y_pred, axis=1)
conf_mat = confusion_matrix(test_data['any'].tolist(), y_pred.tolist())
print('Confusion Matrix')
print(conf_mat, '\n')

# Calculate Accuracy on Test data
print('Accuracy: ', str(round((sum(np.diagonal(conf_mat))/test_data.shape[0])*100, 2))+'%', '\n')

# Classification Report
print('Classification Report')
print(classification_report(test_data['any'].tolist(), y_pred.tolist(), 
                            target_names=['Disease', 'No-Disease']))


# # Prediction on Final Model

# In[12]:


# pass the test data
# Confution Matrix 
Y_pred   = last_model.predict(data.iloc[:,:-5])
Y_pred   = pd.DataFrame(Y_pred)
Y_pred.columns = ['epidural','intraparenchymal', 'intraventricular','subarachnoid','subdural']

print("Predicted")
display(Y_pred)
print("Actual")
display(data.iloc[:,-5:])


# In[ ]:




