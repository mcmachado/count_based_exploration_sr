import tensorflow as tf


def add_simple_summary(writer, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)


def get_size_output_convolution(input, filter, stride, padding=0):
    """
    Compute the size of the image after running through a convolution.
    Args:
            :param input: size of one dimension of the image
            :param filter: size of the filter to run over the input
            :param stride: stride to be used over input
            :param padding: padding to be used over input
    Returns:
            :return: size of output after running through the defined convolution
    """
    size_out = int((input - filter + 2 * padding)/stride + 1)
    return size_out
