from keras.layers import Input, Dense
from keras.models import Model

# building a model
encoding_dim = 32

input_img = Input(shape=(784, ))
encoded = Dense(encoding_dim, activation='relu', name='encoding')(input_img)
decoded = Dense(784, activation='sigmoid', name='decoding')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim, ))
decoded_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoded_layer(encoded_input))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# summariseing the model architecture
autoencoder.summary()

# save the architecture in image
from keras.utils import plot_model
plot_model(autoencoder, to_file='model.png')

# get and prepare dataset
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape, x_test.shape)

# fit the model to the dataset
autoencoder.fit(x_train, x_train, epochs=2, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# saving the model in a format of serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")

# presentation part: try reproducing the image to see how it worked
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# showing the image
import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
	ax = plt.subplot(2, n, i+1)
	plt.imshow(x_test[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# display reconstruction
	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(decoded_imgs[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
