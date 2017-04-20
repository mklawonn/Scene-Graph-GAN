import tensorflow as tf


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.histogram_summary(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    tf.histogram_summary(var.op.name + "/activation", var)
    tf.scalar_summary(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.histogram_summary(var.op.name + "/gradient", grad)
