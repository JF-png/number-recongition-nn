import numpy as np 
import matplotlib.pyplot as plt 
import utils

images, labels = utils.load_dataset()

mustTrain = False

np.set_printoptions(formatter={'float_kind':'{:f}'.format})

if mustTrain:
	weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (70, 784))
	weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 70))

	bias_input_to_hidden = np.zeros((70,1))
	bias_hidden_to_output = np.zeros((10, 1))

	epochs = 3
	e_loss = 0
	e_correct = 0
	lerning_rate = 0.015
	for epoch in range(epochs):
		print(f"Epoch №{epoch}")

		for image, label in zip(images, labels):
			image = np.reshape(image, (-1,1))
			label = np.reshape(label, (-1,1))

			#fprop 
			#hidden
			hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
			hidden = 1/(1+np.exp(-hidden_raw))

			#output
			output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
			output = 1/(1+np.exp(-output_raw))

			# Loss / Error calculation
			e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
			e_correct += int(np.argmax(output) == np.argmax(label))

			#backprop

			delta_output = output - label
			weights_hidden_to_output += -lerning_rate * delta_output @ np.transpose(hidden)
			bias_hidden_to_output += -lerning_rate * delta_output

			delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden *(1 - hidden))
			weights_input_to_hidden += -lerning_rate * delta_hidden @ np.transpose(image)
			bias_input_to_hidden += -lerning_rate * delta_hidden

		# print some debug info between epochs
		print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
		print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
		e_loss = 0
		e_correct = 0
	np.save('weights_input_to_hidden', weights_input_to_hidden)
	np.save('weights_hidden_to_output', weights_hidden_to_output)
	np.save('bias_input_to_hidden', bias_input_to_hidden)
	np.save('bias_hidden_to_output', bias_hidden_to_output)
else:
	weights_input_to_hidden = np.load('weights_input_to_hidden.npy')
	weights_hidden_to_output = np.load('weights_hidden_to_output.npy')

	bias_input_to_hidden = np.load('bias_input_to_hidden.npy')
	bias_hidden_to_output = np.load('bias_hidden_to_output.npy')

#import random
#test_image = random.choice(images)

test_image = plt.imread("custom.jpg", format="jpeg")

# Grayscale + Unit RGB + inverse colors
gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
test_image = 1 - (gray(test_image).astype("float32") / 255)
def predict(test_image):

	image = np.reshape(test_image, (-1, 1))

	hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
	hidden = 1/(1+np.exp(-hidden_raw))

	#output
	output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
	output = 1/(1+np.exp(-output_raw))
	
	print('predictions')
	for i in range(len(output)):
		value = output[i];
		value = str(value)
		value = value.replace('[', '')
		value = value.replace(']', '')
		print(f'chance that number is {i} equals {value}')

	plt.imshow(test_image.reshape(28,28), cmap = "Greys")
	plt.title(f"NN suggests the number is {output.argmax()}")
	plt.show()

predict(test_image)
