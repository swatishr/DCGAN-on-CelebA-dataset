#Citation: https://www.oreilly.com/ideas/deep-convolutional-generative-adversarial-networks-with-tensorflow
#https://arxiv.org/pdf/1511.06434.pdf


import library as lib
import os
from glob import glob
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

#implementing leaky relu
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

#to create model inputs for real data and one noise data
def model_inputs(img_width, img_height, img_channels, z_dim):
	real_inputs = tf.placeholder(tf.float32, shape=(None, img_width, img_height, img_channels), name='real_input')
	noise_input = tf.placeholder(tf.float32,shape=(None, z_dim), name='z_input')
	learning_rate = tf.placeholder(tf.float32, name='learning_rate')
	return real_inputs, noise_input, learning_rate

#build the discriminator network which is a CNN with 3 convolutional layers (original paper has 4 layers)
#perform convolution, batch normalization to make network faster & more accurate and then perform ReLU
#at the last layer, apply sigmoid to classify the image as real or fake
def discriminator(images, reuse=False):
	alpha = 0.2

	with tf.variable_scope('discriminator', reuse=reuse):
		# using 4 layer network as in DCGAN Paper

		# Conv 1
		conv1 = tf.layers.conv2d(images, 64, 5, 2, 'SAME')
		lrelu1 = lrelu(conv1)

		# Conv 2
		conv2 = tf.layers.conv2d(lrelu1, 128, 5, 2, 'SAME')
		batch_norm2 = tf.layers.batch_normalization(conv2, training=True)
		lrelu2 = lrelu(batch_norm2)

		# Conv 3
		conv3 = tf.layers.conv2d(lrelu2, 256, 5, 1, 'SAME')
		batch_norm3 = tf.layers.batch_normalization(conv3, training=True)
		lrelu3 = lrelu(batch_norm3)

		# Flatten
		flat = tf.reshape(lrelu3, (-1, 4*4*256))

		# Logits
		logits = tf.layers.dense(flat, 1)

		# Output
		out = tf.sigmoid(logits)

		return out, logits


#Generator network that is nuild using deconvolutional NN having 3 deconv layers 
#and at last tanh layer to generate the image from the given noise vector
def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    """
    alpha = 0.2
    
    with tf.variable_scope('generator', reuse=False if is_train==True else True):
        # First fully connected layer
        x_1 = tf.layers.dense(z, 2*2*512)
        
        # Reshape it to start the convolutional stack
        deconv_2 = tf.reshape(x_1, (-1, 2, 2, 512))
        batch_norm2 = tf.layers.batch_normalization(deconv_2, training=is_train)
        #lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)
        lrelu2 = tf.nn.relu(batch_norm2)

        # Deconv 1
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 256, 5, 2, padding='VALID')
        batch_norm3 = tf.layers.batch_normalization(deconv3, training=is_train)
        #lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)
        lrelu3 = tf.nn.relu(batch_norm3)
        
        # Deconv 2
        deconv4 = tf.layers.conv2d_transpose(lrelu3, 128, 5, 2, padding='SAME')
        batch_norm4 = tf.layers.batch_normalization(deconv4, training=is_train)
        #lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)
        lrelu4 = tf.nn.relu(batch_norm4)

        # Output layer
        logits = tf.layers.conv2d_transpose(lrelu4, out_channel_dim, 5, 2, padding='SAME')
        
        out = tf.tanh(logits)
        
        return out

#Loss function
#There are in total 3 losses: 1st for generator network, 2nd for discriminator when using real images, 
#3rd for discriminator when using fake images; the overall loss for discriminator is sum of 2nd and 3rd
def model_loss(real_input, z_input, out_channel_dim):
	label_smoothing = 0.9

	g_model = generator(z_input, out_channel_dim)
	d_model_real, d_logits_real = discriminator(real_input)
	d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_model_real)*label_smoothing))
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_model_fake)*label_smoothing))

	d_loss = d_loss_real + d_loss_fake

	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_model_fake) * label_smoothing))

	return g_loss, d_loss

#Optimization
def model_optimize(d_loss, g_loss, learning_rate, beta1):
	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
	g_vars = [var for var in t_vars if var.name.startswith('generator')]

	# Optimize
	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
		d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
		g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

	return d_train_opt, g_train_opt

#helper function to display the generator output
def show_generator_output(sess, n_images, input_z, out_channel_dim):
	z_dim = input_z.get_shape().as_list()[-1]
	example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

	samples = sess.run(generator(input_z, out_channel_dim, False), feed_dict={input_z: example_z})

	plt.imshow(lib.images_square_grid(samples))
	plt.show()

#training the network : beta1 is the momentum
def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape):
	real_input, z_input, _ = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
	g_loss, d_loss = model_loss(real_input, z_input, data_shape[3])
	d_opt, g_opt = model_optimize(d_loss, g_loss, learning_rate, beta1)

	steps = 0

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(epoch_count):
			for batch_images in lib.get_batches(batch_size, data_shape, img_files):
				batch_images = batch_images * 2 #changing scale to -1 to 1

				if (steps%10) == 0:
					print("Step: {}".format(steps))

				steps+=1

				batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
				#??????
				_ = sess.run(d_opt, feed_dict={real_input: batch_images, z_input: batch_z})
				_ = sess.run(g_opt, feed_dict={real_input: batch_images, z_input: batch_z})

				if(steps%400) == 0:
					train_loss_d = d_loss.eval({z_input: batch_z, real_input: batch_images})
					train_loss_g = g_loss.eval({z_input: batch_z})
					print("Epoch {}/{}...".format(epoch+1, epochs),"Discriminator Loss: {:.4f}...".format(train_loss_d),"Generator Loss: {:.4f}".format(train_loss_g))

					_ = show_generator_output(sess, 1, z_input, data_shape[3])


#Download the dataset
lib.download_celeb_a()

data_dir = './data'
data_file_name = 'celebA/*.jpg'
#Image configuration
IMG_HEIGHT = 28;
IMG_WIDTH = 28;
img_files = glob(os.path.join(data_dir,data_file_name))
data_shape = len(img_files), IMG_WIDTH, IMG_HEIGHT, 3 #3 for RGB

test_images = lib.get_batch((img_files)[:10], 56, 56)
plt.figure()
plt.imshow(lib.images_square_grid(test_images))
plt.show()

batch_size = 16
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5
epochs = 2

with tf.Graph().as_default():
	train(epochs, batch_size, z_dim, learning_rate, beta1, lib.get_batches, data_shape)