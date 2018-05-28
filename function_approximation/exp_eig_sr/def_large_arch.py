# Author Marlos C. Machado


def compute_size_output_convolution(input_sz, filter_sz, stride_sz, padding_sz=0):
    size_output = int((input_sz - filter_sz + 2 * padding_sz)/stride_sz + 1)
    return size_output


# General parameters
input_height = 84
input_width = 84
num_actions = 18
frame_history_size = 4

# Representation learning module:
stride_conv1 = 2
padding_conv1 = [0, 0]
num_filters_conv1 = 64
kernel_size_conv1 = [6, 6]

stride_conv2 = 2
padding_conv2 = [2, 2]
num_filters_conv2 = 64
kernel_size_conv2 = [6, 6]

stride_conv3 = 2
padding_conv3 = [2, 2]
num_filters_conv3 = 64
kernel_size_conv3 = [6, 6]

num_nodes_fc_representation = 1024

# Reconstruction module
num_nodes_fc_in_reconstruction = 2048
num_nodes_fc_action_embedding = 2048
num_nodes_fc1_out_reconstruction = 1024
num_nodes_fc2_out_reconstruction = 6400

# It is easier to do it backwards

shape_input_deconv3 = [compute_size_output_convolution(input_height, kernel_size_conv1[0], stride_conv1, padding_conv1[0]),
                       compute_size_output_convolution(input_width, kernel_size_conv1[1], stride_conv1, padding_conv1[1])]

shape_input_deconv2 = [compute_size_output_convolution(shape_input_deconv3[0], kernel_size_conv2[0], stride_conv2, padding_conv2[0]),
                       compute_size_output_convolution(shape_input_deconv3[1], kernel_size_conv2[1], stride_conv2, padding_conv2[1])]

shape_input_deconv1 = [compute_size_output_convolution(shape_input_deconv2[0], kernel_size_conv3[0], stride_conv3, padding_conv3[0]),
                       compute_size_output_convolution(shape_input_deconv2[1], kernel_size_conv3[1], stride_conv3, padding_conv3[1])]

stride_deconv1 = 2
padding_deconv1 = [2, 2]
num_filters_deconv1 = 64
kernel_size_deconv1 = [6, 6]

stride_deconv2 = 2
padding_deconv2 = [2, 2]
num_filters_deconv2 = 64
kernel_size_deconv2 = [6, 6]

stride_deconv3 = 2
padding_deconv3 = [0, 0]
num_filters_deconv3 = 1
kernel_size_deconv3 = [6, 6]

# SR module
num_nodes_fc1_sr = 2048
num_nodes_fc2_sr = 1024
