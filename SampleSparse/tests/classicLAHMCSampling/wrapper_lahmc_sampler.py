#Wrapper Script to run the lahmc_code with various paramters
#Implemented by Doron Reuven

from lahmc_sparse import lahmc_sampler
from BrunoSparseCodingTensorflow_for_testing import TensorSparse
from scipy import io
import numpy as np
import random
import os
import matplotlib.pyplot as  plt
import ipdb
import tensorflow as tf
from BrunoSparseCodingTensorflow_for_testing import TensorSparse
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
batch_size = 200
#change above back to 5
lambda_parameter = 1e-1
LR = 1e-1
border = 4
patch_dim = size_of_patch**2
sz = np.sqrt(patch_dim)
num_particles_per_image = 10
#change above back to 100
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
#####################
#This code below is sparse coding trained with the same parameters as above and it its purpose is the pickled phi data
#toggle the below to run
def run_sparse_coding(num_receptive_fields, size_of_patch, batch_size, lambda_parameter, LR):
    if not os.path.exists("Bruno_Phi_Pickle"):
        os.mkdir("Bruno_Phi_Pickle")
    training_iterations = 1500
    plot_directory = 'Plots_Bruno_Sparse_Coding'
    sess = tf.Session()
    our_class = TensorSparse(num_receptive_fields = num_receptive_fields, size_of_patch = size_of_patch,
                         session_object = sess, batch_size = batch_size, lambda_parameter = lambda_parameter,
                         LR = LR, plot_directory = plot_directory)
    for i in range(training_iterations):
        print("On iter", i)
        our_data = get_batch_im(our_images, num_images)
        our_class.load_data(our_data)
        our_class.infer_a_coefficients(i)
        our_class.update_phis(i, "Bruno_Phi_Pickle")
        if i % 100 == 0:
            our_phis = our_class.phis_so_far
            our_class.plot_obj.plot_input_data(our_data, i)
            our_class.plot_obj.plot_phis(our_phis, i)
########
#Toggle the below to run sparse coding to recieve the basis


# run_sparse_coding(num_receptive_fields, size_of_patch, batch_size, lambda_parameter, LR)


#######
#load pickle file from
##Initliaze class###
tf.Session().close()
sess = tf.Session()
lahmc_class = lahmc_sampler(num_receptive_fields,size_of_patch,batch_size,lambda_parameter,session_object = sess, num_particles_per_batch = num_particles_per_image)
####
for ii in range(300):
    print("\n\n On iter {0} \n".format(ii))
    batch_data = get_batch_im(our_images,num_images)
    lahmc_class.load_batch(batch_data)
    lahmc_class.sample(ii,num_steps = 1)
print("finished!")
tf.Session().close()
ipdb.set_trace()
    #change above to 1000
#Extracting results from class for plott
# result_energies = lahmc_class.ret_ze_sample_energies()
# print("LAHMC energies are", result_energies)
# all_points = []
# for list_pts in result_energies:
#     for pt in list_pts:
#         all_points.append(pt)
# x = np.arange(len(all_points))
# plt.plot(x,all_points, 'ro')
# ipdb.set_trace()
# plt.show()

# plt.plot(result_energies)
# plt.show()

# ###Initliazing original sparse coding model for comparison###
# phi_lahmc = lahmc_class.phis
# bruno_class = TensorSparse(num_receptive_fields = num_receptive_fields, size_of_patch = size_of_patch, batch_size = batch_size,lambda_parameter = lambda_parameter, session_object = sess, LR = 1e-1, plot_directory = 'Bruno_2_comp_LAHMC', phis = phi_lahmc)

# bruno_class.load_data(batch_data)
# bruno_class.infer_a_coefficients(sess, 0)
# #Now that we have the a valuesplot the energy
# energy_vals = bruno_class.energy_function_for_plotting()
# print(energy_vals)
