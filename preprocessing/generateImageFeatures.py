import tensorflow as tf
import os
import utils
import numpy as np

#############################################################
# Takes in the path to a saved tensorflow graph definition
# and loads it into memory
#############################################################
def loadModel(path_to_model):
    with open(path_to_model, mode='rb') as f:
        file_content = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file_content)

    return graph_def 


#############################################################
# Takes a VGG-16 graph definition and path to an image
# and returns the features of that image
#############################################################
def featuresForImage(graph_definition, path_to_image):
    images = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(graph_definition, input_map={"images":images})
    graph = tf.get_default_graph()
    img = utils.load_image(path_to_image)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        batch = img.reshape((1, 224, 224, 3))

        feed_dict = {images : batch}
        
        feature_tensor = graph.get_tensor_by_name("import/Relu1")
        features = sess.run(feature_tensor, feed_dict=feed_dict)
    return features



#############################################################
# Takes a VGG-16 graph definition and path to an image
# and returns the features of that image
#############################################################
def featuresForAllImages(path_to_images, path_to_model):
    graph_def = loadModel(path_to_model)
    imgs = []
    r_mean, g_mean, b_mean = utils.compute_mean_for_images(path_to_images)
    for image in os.listdir(path_to_images):
        #Read in all images in dir
        img = utils.load_image(image)
        #Subtract image mean from image
        img[:,:,0] -= r_mean
        img[:,:,1] -= g_mean
        img[:,:,2] -= b_mean
        img = img.reshape((224, 224, 3))
        imgs.append(img)
    #Construct feed graph for all images in dir
    images = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={"images":images})
    graph = tf.get_default_graph()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        feed_dict = {images : np.asarray(imgs)}

        feature_tensor = graph.get_tensor_by_name("import/Relu1")
        features = sess.run(feature_tensor, feed_dict=feed_dict)
    #Save tensors
    np.save("features.npz", features)
