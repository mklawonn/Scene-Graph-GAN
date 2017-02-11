import tensorflow as tf

def loadModel(path_to_model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
