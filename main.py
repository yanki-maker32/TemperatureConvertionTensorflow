#main model runs through this formula where
#Fahrenheit = Celsius * 1.8 + 32
#Of course, it would be simple enough to create a conventional Python
# function that directly performs this conversion, but that would not be
#machine learning.

#Instead, we’ll give TensorFlow some sample Celsius values and their corresponding
#Fahrenheit values. Then, we’ll train a model that figures out the above formula
#through the training process. And we’d use the model to convert arbitrary values
#from Celsius to Fahrenheit and finally evaluate the model.


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Let's create two lists for celsius and fahrenheit using numpy
#np to create array dimensions
celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)
# Let's iterate through the list and print out the corresponding #values
for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

#For simplicity’s sake, this network will have only one layer and one neuron
#layer1#
#unit is neuron, layers are shape
#L1 = tf.keras.layers.Dense(units=1, input_shape=[1])
#tf.keras.Sequential([L1])
#keras is an API specification that describes how a Deep Learning framework should implement
# certain part, related to the model definition and training.
# Is framework agnostic and supports different backends
#Sequential groups a linear stack of layers into a tf.keras.Model.
#Sequential provides training and inference features on this model.
#Takes layers as list
model = tf.keras.Sequential(tf.keras.layers.Dense(units=1, input_shape=[1]))
#enhances tf's ability to perform and train models
#1. Loss Function: A way of measuring the difference between predictions and the desired outcome.
#2. Optimizer Function: A way of adjusting internal values or parameters in a bid to reduce the loss.
model.compile(loss ='mean_squared_error', optimizer= tf.keras.optimizers.Adam(0.1))
#Adam is our learning rate algorithm
#We train the model by calling the fit() method.
#Once the model is created, you can config the model with losses and metrics with model.compile()
#, train the model with model.fit(), or use the model to do prediction with model.predict().
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=True)
print("Finished training the model")

plt.xlabel('Epoch Number', color='black')
plt.ylabel("Loss Magnitude", color='black')
plt.plot(history.history['loss'])
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

def test_model(model):
    MSE = []
    xx, yy = [], []
    for x in range(10, 210, 10):
        y_hat = model.predict([x]).astype(float)
        y_hat = y_hat[0]
        y = x * 1.8 + 32
        error_squ = (y_hat - y)**2
        MSE.append(error_squ)
        print('celsius is {}, Fahrenheit is {}, Model predicted Fahrenheit is {}, Diff_Squared is {}'.format(x, y, y_hat, error_squ ))
    MSE = sum(MSE) / len(MSE)
    print('Total MSE is {}'.format(MSE))
print(test_model(model))