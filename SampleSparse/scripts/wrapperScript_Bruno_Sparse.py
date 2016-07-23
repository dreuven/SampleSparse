import tensorflow as tf
from BrunoSparseCodingTensorflow import TensorSparse
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Create class parameters
num_receptive_fields = 1024
size_of_patch = 32
batch_size = 200
lambda_parameter = .1
LR = 1e-1
training_iterations = 2500
border = 4
plot_directory = 'Plots_tensor'
# Instantiate class
sess = tf.Session()
our_class = TensorSparse(num_receptive_fields = num_receptive_fields, size_of_patch = size_of_patch,
    session_object = sess, batch_size = batch_size, lambda_parameter = lambda_parameter, 
    LR = LR, plot_directory = plot_directory)

init = tf.initialize_all_variables()
sess.run(init)
data_mayur = np.zeros((size_of_patch**2.0,batch_size))
patch_dim = size_of_patch**2.0
sz = np.sqrt(patch_dim)
# Training set
DATA = os.getenv('DATA')
proj_path = DATA + 'SampleSparse/'
data_path = proj_path + 'Bruno_Whitened_Natural_Scenes.mat'
our_images = io.loadmat(data_path)["IMAGES"]
(imsize, imsize,num_images) = np.shape(our_images)
for i in range(training_iterations):
    print("On iter {}".format(i))
    # generating samples my way    
    # the_image_index = int(np.random.randint(0,10,1)[0])
    # input_image = our_images[:,:,the_image_index]
    for j in range(batch_size):
        #Moving the image choosing inside the loop, so we get more randomness in image choice
          imi = np.ceil(num_images * random.uniform(0, 1))
          r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
          c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
          data_mayur[:,j] = np.reshape(our_images[r:r+sz, c:c+sz, imi-1], patch_dim, 1)
    # data = get_n_patches_from_image_of_certain_size(input_image, size_of_patch, batch_size)
    our_class.load_data(data_mayur)
    our_class.infer_a_coefficients(sess, i)
    our_class.update_phis()
    our_class.handle_plotting_in_tensorboard(sess, i)
    plt.figure()
    plt.title("Recon Error")
    plt.plot(our_class.reconstruction_error_array)
    plt.close()
    if i % 100 == 0:
        # feed_dict = {our_class.data : data}
        our_phis = sess.run(our_class.phis)
        our_class.plot_obj.plot_input_data(data_mayur, i)
        our_class.plot_obj.plot_phis(our_phis, i)
        # our_class.plo
