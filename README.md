# TemperatureConvertionTensorflow
Using basic tensorflow in built keras functions, I have created a module to convert fahrenheit values into Celsius, model training, and pre built data frame
that includes fahrenheit values are given inside the py module. Instead of building multiple layers, our code chunk takes the numpy arrays from preset values, and converts them
some what accurately into celcius values. Success rate can be increased using a larger csv data set and using multiple layers. 

This project was made by Yanki Saplan, and instructed to listeners through Penn State HACK series, to more than 50 students, through several colleges via zoom presentation
under Google Developer Club at Penn State. 

1) main model runs through this formula where Fahrenheit = Celsius * 1.8 + 32
 - Of course, it would be simple enough to create a conventional Python
 - function that directly performs this conversion, but that would not be machine learning.

2) Instead, we’ll give TensorFlow some sample Celsius values and their corresponding
 - Fahrenheit values. Then, we’ll train a model that figures out the above formula
 - through the training process. And we’d use the model to convert arbitrary values
 - from Celsius to Fahrenheit and finally evaluate the model.
 
 3) For simplicity’s sake, this network will have only one layer and one neuron #layer1#
    #unit is neuron, layers are shape
    #L1 = tf.keras.layers.Dense(units=1, input_shape=[1])
    #tf.keras.Sequential([L1])
    #keras is an API specification that describes how a Deep Learning framework should implement
    certain part, related to the model definition and training.
    Is framework agnostic and supports different backends
    #Sequential groups a linear stack of layers into a tf.keras.Model.
    #Sequential provides training and inference features on this model.
    #Takes layers as list
 
 4) Loss Function: A way of measuring the difference between predictions and the desired outcome.
    Optimizer Function: A way of adjusting internal values or parameters in a bid to reduce the loss.
