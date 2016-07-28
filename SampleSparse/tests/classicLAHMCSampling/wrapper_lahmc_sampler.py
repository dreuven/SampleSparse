from tflow_lahmc_class_testing import lahmc_sampler
from BrunoSparseCodingTensorflow_for_testing import TensorSparse
import tensorflow as tf
from scipy import io
import numpy as np
import random
import os
import matplotlib.pyplot as  plt
#Load images
DATA = os.getenv('DATA')
proj_path = DATA + 'SampleSparse/'
data_path = proj_path + 'Bruno_Whitened_Natural_Scenes.mat'
our_images = io.loadmat(data_path)["IMAGES"]
(imsize, imsize,num_images) = np.shape(our_images)
############

##Parameters##
num_receptive_fields = 128
size_of_patch = 10
batch_size = 1
lambda_parameter = 1e-1
LR = 1e-1
border = 4
patch_dim = size_of_patch**2
sz = np.sqrt(patch_dim)
num_particles_per_batch = 100
##############


def get_batch_im(our_images,num_images):
    batch_data = np.zeros((size_of_patch**2,batch_size))

    for j in range(batch_size):
        #Moving the image choosing inside the loop, so we get more randomness in image choice
        imi = np.ceil(num_images * random.uniform(0, 1))
        r = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
        c = border + np.ceil((imsize-sz-2*border) * random.uniform(0, 1))
        batch_data[:,j] = np.reshape(our_images[r:r+sz, c:c+sz, imi-1], patch_dim, 1)
    return batch_data

batch_data = get_batch_im(our_images,num_images)

##Initliaze class###
sess = tf.Session()
lahmc_class = lahmc_sampler(num_receptive_fields,size_of_patch,batch_size,lambda_parameter,session_object = sess, num_particles_per_batch = num_particles_per_batch)
####
lahmc_class.load_batch(batch_data)
lahmc_class.sample(20)
#Extracting results from class for plott
result_energies = lahmc_class.ret_ze_sample_energies()
print("LAHMC energies are", result_energies)
plt.plot(result_energies)
plt.show()

###Initliazing original sparse coding model for comparison###
phi_lahmc = lahmc_class.phis
bruno_class = TensorSparse(num_receptive_fields = num_receptive_fields, size_of_patch = size_of_patch, batch_size = batch_size,lambda_parameter = lambda_parameter, session_object = sess, LR = 1e-1, plot_directory = 'Bruno_2_comp_LAHMC', phis = phi_lahmc)

bruno_class.load_data(batch_data)
bruno_class.infer_a_coefficients(sess, 0)
#Now that we have the a valuesplot the energy
energy_vals = bruno_class.energy_function_for_plotting()
print(energy_vals)
