#Import all libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras

#Get The Data
df = pd.read_csv('Cancer_Study.csv',index_col = None)
df.drop(columns = ['Unnamed: 32','id'],inplace = True)
df.replace('M','1',inplace = True)
df.replace('B','0',inplace =True)

#Create X and Y
X = df.iloc[:,1:31]
y = df.iloc[:,0]
X = (X-np.mean(X))/np.var(X)


#Split test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.2, random_state = 1)

model = keras.Sequential([
    keras.layers.Dropout(0.25, input_shape=(30,)),
    keras.layers.Dense(2**8,activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(2**7,activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(2**6,activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(2**5,activation = 'relu'),
    keras.layers.BatchNormalization(),
    
    keras.layers.Dense(1,activation = 'sigmoid')])

opt = keras.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer = opt,
              loss ='binary_crossentropy',
              metrics = ['accuracy'])

yhat = model.predict(X_test)
scores2 = model.evaluate(X_test, y_test)
scores = model.evaluate(X_train, y_train)

history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=500, batch_size=20)
scores2 = model.evaluate(X_test, y_test)
scores = model.evaluate(X_train, y_train)
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()