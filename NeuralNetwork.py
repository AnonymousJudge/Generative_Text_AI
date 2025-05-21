
import numpy
from helper.JsonWriter import write_list_to_JSON, load_list_from_JSON
from structures.Errors import Layer_out_of_bounds

class NeuralNetwork:
	
	def __init__(self, path, layers, alpha=0.003):
		# initialize the list of weights matrices, then store the
		# network architecture and learning rate
		self.W = []
		self.layers = layers
		self.alpha = alpha
		self.w_path = path
		if path is not None:
			try:
				self.import_weights()
			except:
				self.gen_weights()
		else: 
			self.gen_weights()

	def  __repr__(self):
		# construct and return a string that represents the network
		# architecture
		return "NeuralNetwork: {}".format(
			"-".join(str(l) for l in self.layers))
	
	def gen_weights(self):
		# start looping from the index of the first layer but
		# stop before we reach the last two layers
		for i in numpy.arange(0, len(self.layers) - 2):
			# randomly initialize a weight matrix connecting the
			# number of nodes in each respective layer together,
			# adding an extra node for the bias
			w = numpy.random.randn(self.layers[i] + 1, self.layers[i + 1] + 1)
			self.W.append(w / numpy.sqrt(self.layers[i]))
			
		# the last two layers are a special case where the input
		# connections need a bias term but the output does not
		w = numpy.random.randn(self.layers[-2] + 1, self.layers[-1])
		self.W.append(w / numpy.sqrt(self.layers[-2]))

	def import_weights(self):
		data = load_list_from_JSON(self.w_path)
		for d in data:
			self.W.append(numpy.array([numpy.array(li) for li in d]))

	def export_weights(self):
		write_list_to_JSON(self.W, self.w_path)

	def sigmoid(self, x: numpy.ndarray) -> numpy.ndarray:
		"""
        Compute and return the sigmoid activation value for a given input array or value.
        
        :param x: Input value or array (numpy.ndarray).
        :return: Sigmoid activation value (numpy.ndarray or float).
        """
		return 1.0 / (1 + numpy.exp(-x))
	
	def sigmoid_deriv(self, x):
		# compute the derivative of the sigmoid function ASSUMING
		# that x has already been passed through the 'sigmoid'
		# function
		return x * (1 - x)
	
	def fit(self, X:numpy.ndarray, y:numpy.ndarray, epochs=1000, displayUpdate=100):
		"""
		Parameters:
			X: training data
			y: correspondin class labels
		"""
		# insert a column of 1's as the last entry in the feature
		# matrix -- this little trick allows us to treat the bias
		# as a trainable parameter within the weight matrix
		X = numpy.c_[X, numpy.ones((X.shape[0]))]
		# loop over the desired number of epochs
		for epoch in numpy.arange(0, epochs):
			# loop over each individual data point and train
			# our network on it
			for (x, target) in zip(X, y):
				self.fit_partial(x, target)
			# check to see if we should display a training update
			if epoch == 0 or (epoch + 1) % displayUpdate == 0:
				loss = self.calculate_loss(X, y)
				print("[INFO] epoch={}, loss={:.7f}".format(
					epoch + 1, loss))
				
	def fit_partial(self, x, y):
		"""
		Parameters:
			x: An individual data point from our design matrix.
			y: The corresponding class label.
		Returns:

		"""
		# construct our list of output activations for each layer
		# as our data point flows through the network; the first
		# activation is a special case -- it's just the input
		# feature vector itself
		A = [numpy.atleast_2d(x)]

		# FEEDFORWARD:
		# loop over the layers in the network
		for layer in numpy.arange(0, len(self.W)):
			# feedforward the activation at the current layer by
			# taking the dot product between the activation and
			# the weight matrix -- this is called the "net input"
			# to the current layer
			net = A[layer].dot(self.W[layer])
			# computing the "net output" is simply applying our
			# nonlinear activation function to the net input
			out = self.sigmoid(net)
			# once we have the net output, add it to our list of
			# activations
			A.append(out)
	
		# BACKPROPAGATION
		# the first phase of backpropagation is to compute the
		# difference between our *prediction* (the final output
		# activation in the activations list) and the true target
		# value
		error = A[-1] - y
		# from here, we need to apply the chain rule and build our
		# list of deltas 'D'; the first entry in the deltas is
		# simply the error of the output layer times the derivative
		# of our activation function for the output value
		D = [error * self.sigmoid_deriv(A[-1])]

		# once you understand the chain rule it becomes super easy
		# to implement with a 'for' loop -- simply loop over the
		# layers in reverse order (ignoring the last two since we
		# already have taken them into account)
		for layer in numpy.arange(len(A) - 2, 0, -1):
			# the delta for the current layer is equal to the delta
			# of the *previous layer* dotted with the weight matrix
			# of the current layer, followed by multiplying the delta
			# by the derivative of the nonlinear activation function
			# for the activations of the current layer
			delta = D[-1].dot(self.W[layer].T)
			delta = delta * self.sigmoid_deriv(A[layer])
			D.append(delta)

		# since we looped over our layers in reverse order we need to
		# reverse the deltas
		D = D[::-1]
		# WEIGHT UPDATE PHASE
		# loop over the layers
		for layer in numpy.arange(0, len(self.W)):
			# update our weights by taking the dot product of the layer
			# activations with their respective deltas, then multiplying
			# this value by some small learning rate and adding to our
			# weight matrix -- this is where the actual "learning" takes
			# place
			self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

	def predict(self, X, addBias=True):
		"""
		Parameters:
			X: The data points we'll be predicting class labels for.
			addBias: A boolean indicating whether we need to add a column of 1's to X to perform the bias trick. 
		"""
		# initialize the output prediction as the input features -- this
		# value will be (forward) propagated through the network to
		# obtain the final prediction
		p = numpy.atleast_2d(X)
		# check to see if the bias column should be added
		if addBias:
			# insert a column of 1's as the last entry in the feature
			# matrix (bias)
			p = numpy.c_[p, numpy.ones((p.shape[0]))]
		# loop over our layers in the network
		for layer in numpy.arange(0, len(self.W)):
			# computing the output prediction is as simple as taking
			# the dot product between the current activation value 'p'
			# and the weight matrix associated with the current layer,
			# then passing this value through a nonlinear activation
			# function
			p = self.sigmoid(numpy.dot(p, self.W[layer]))
		# return the predicted value
		return p
	
	def calculate_loss(self, X, targets):
		# make predictions for the input data points then compute
		# the loss
		targets = numpy.atleast_2d(targets)
		predictions = self.predict(X, addBias=False)
		loss = 0.5 * numpy.sum((predictions - targets) ** 2)
		# return the loss
		return loss
	
	def input_to_vector(self, X:list[int], v_layer:int, addBias= True) -> numpy.ndarray:
		
		if v_layer > len(self.W):
			raise Layer_out_of_bounds()
		
		vector = numpy.atleast_2d(X)
		# check to see if the bias column should be added
		if addBias:
			# insert a column of 1's as the last entry in the feature
			# matrix (bias)
			vector = numpy.c_[vector, numpy.ones((vector.shape[0]))]
		
		for layer in numpy.arange(0, v_layer):
			vector = self.sigmoid(numpy.dot(vector, self.W[layer]))
		
		return vector