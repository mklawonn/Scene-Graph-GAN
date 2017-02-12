import tensorflow as tf
import os
import utils

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

#TODO Replace featuresForImage with this function
def featuresForAllImages(path_to images, path_to_model):
    graph_def = loadModel(path_to_model)
    for image in os.listdir(path_to_images):
        #TODO read in all images in dir
        #TODO preprocess all images in dir
        #TODO construct feed graph for all images in dir
        #TODO Compute feature tensors for all
        #TODO Save tensors
