from numpy_lahmc import numpy_sampler
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
batch_size = 500
lambda_parameter = 1e-1
LR = 1e-1
border = 4
patch_dim = size_of_patch**2
sz = np.sqrt(patch_dim)
num_particles_per_batch = 2
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

our_class = numpy_sampler(num_receptive_fields = num_receptive_fields, size_of_patch = size_of_patch, batch_size = batch_size,lambda_parameter = lambda_parameter, num_particles_per_batch = num_particles_per_batch, LR = LR)

for _ in range(3):
    print("Starting new update! We are on iter {0}\n".format(_))
    batch_data = get_batch_im(our_images, num_images)
    our_class.load_batch(batch_data)
    A_final = our_class.sample(num_steps = 1000)
    print("A final is", A_final)
    result_energies = our_class.ret_sample_energies()
    print("Result energy is", result_energies)
