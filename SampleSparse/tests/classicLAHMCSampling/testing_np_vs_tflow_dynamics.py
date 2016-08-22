import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from LAHMC.python import LAHMC
import ipdb
## Set the seed from which life springs deterministically###
np.random.seed(1234)
#####
np_energy_vals = []
np_deriv_vals = []
np_energy_arg = []
np_deriv_arg = []
def E(X, sigma = 1.):
    # print("Called np energy function!!!!!!!")
    np_energy_arg.append(X)
    energy = np.sum(X**2, axis = 0).reshape(1,-1)/2./sigma**2
    np_energy_vals.append(energy)
    return energy
def dedx(X, sigma = 1.):
    np_deriv_arg.append(X)
    deriv = X/sigma ** 2
    np_deriv_vals.append(deriv)
    return deriv
### Tensorflow implementation
sess = tf.Session()
input = tf.placeholder("float64",shape = (2,None))
energy = tf.reduce_sum(input**2, reduction_indices = 0)/2.
e_grad = tf.gradients(energy,[input])[0]
sess.run(tf.initialize_all_variables())

tflow_energy_vals = []
tflow_deriv_vals = []
tflow_energy_arg = []
tflow_deriv_arg = []
def E_tflow(X, sigma = 1):

    # print("Called tflow energy func")
    # print("In E_tflow and arg is {0}".format(X))
    tflow_energy_arg.append(X)
    energy_ = sess.run(energy, feed_dict = {input:X}).reshape(1,-1)
    # print("In tensorflow energy func and tflow energy is {0}".format(energy_))
    tflow_energy_vals.append(energy_)
    return energy_
def dedx_flow(X, sigma = 1):
    tflow_deriv_arg.append(X)
    deriv = sess.run(e_grad, feed_dict = {input: X})
    # print("In tensorflow derivative and it is {0}".format(deriv))
    tflow_deriv_vals.append(deriv)
    return deriv

#Lahmc params and implementation
epsilon = 0.1
beta = 0.2
num_look_ahead_steps = 1
num_leapfrog_steps = 1
#Arrays for comparison
np_sample_array = []
tflow_sample_array = []
for i in range(1):
    Ainit = np.random.random((2,100))
    # x = Ainit.copy()
    # print(Ainit)
    #Numpy sample
    # print("Now handling Numpy Version: \n")
    print("Calling np sampler")
    print("\n\n")
    sampler_np = LAHMC.LAHMC(Ainit.copy(),E, dedx, epsilon=epsilon, beta = beta, num_look_ahead_steps=10, num_leapfrog_steps = 10)
    A_final_np = sampler_np.sample(100)
    np_sample_array.append(A_final_np)
    #tflow sample
    # print("\n\nNow handling Tflow sample: \n\n")
    print("Calling tflow sampler")
    # print(Ainit)
    np.random.seed(1234)
    Ainit = np.random.random((2,100))
    # print(x == Ainit)

    sampler_tflow =  LAHMC.LAHMC(Ainit.copy(),E_tflow, dedx_flow, epsilon=epsilon, beta=beta, num_look_ahead_steps=10, num_leapfrog_steps = 10)
    # sampler_tflow = LAHMC.LAHMC(Ainit.copy(),E, dedx, epsilon=epsilon, beta = beta, num_look_ahead_steps=num_look_ahead_steps, num_leapfrog_steps = 10)
    A_final_tflow = sampler_tflow.sample(100)
    tflow_sample_array.append(A_final_tflow)

np_pairs = zip(np_energy_vals,np_deriv_vals)
tflow_pairs = zip(tflow_energy_vals, tflow_deriv_vals)

# print("\n\n Beginning analysis\n\n")

# for np_arg,tflow_arg in zip(np_energy_arg,tflow_energy_arg):
#     print("Np argument is {0} \n tflow arg is {1} \n\n--------------------------\n".format(np_arg, tflow_arg))


count = 0
for a,b in zip(np_pairs,tflow_pairs):
    np_e = a[0]
    np_d = a[1]
    t_e = b[0]
    t_d = b[1]
    print("Energy comparison")
    print(np_e == t_e, "\n\n\n")
    print("Derivative comparison\n\n")
    print(np_d == t_d)
    print()
    print("On iteration {0} np energy value is {1} \n tflow_energy_value is {2} \n np_deriv is {3} \n tflow_deriv {4}\n-------------------\n".format(count, np_e,t_e,np_d,t_d))
    count += 1
##Shittily compare by inspection
# count = 0
# for a,b in zip(np_sample_array,tflow_sample_array):
#     print("\n-----------------------------/n")
#     print("For count {0} np_sample is {1}\n\n while tflow_sample is {2} \n\n".format(count, a, b))
#     count += 1
