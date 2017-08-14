import os, sys
sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
import argparse
import json

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from train import SceneGraphWGAN

#####################################################
# Global Variables
#####################################################

def loadModel(params, sess):
    #Create WGAN instance
    wgan = SceneGraphWGAN(params["vg_batches"], params["vocab"], params["generator"], params["discriminator"], params["logs_dir"], params["samples_dir"], 
           BATCH_SIZE=params["batch_size"], CRITIC_ITERS=params["critic_iters"], LAMBDA=params["lambda"], im_and_lang=params["use_language"],
           validation = True, resume=False, dataset_relations_only = params["dataset_relations_only"])
    wgan.constructOps()
    wgan.loadModel(sess)
    queue_var_name = wgan.queue_var_name
    #return tf.get_default_graph().as_graph_def()
    tf.train.start_queue_runners(sess=sess)
    wgan.custom_runner.start_threads(sess)
    return wgan

def generatePredictions(wgan, sess):
    #Initialize the queue variables
    queue_vars = [v for v in tf.global_variables() if wgan.queue_var_name in v.name]
    queue_init_op = tf.variables_initializer(queue_vars)
    sess.run(queue_init_op)

    gen_output, disc_cost, subject_att, predicate_att, object_att = sess.run([wgan.fake_inputs, wgan.disc_cost, wgan.g.attention_vectors[0], wgan.g.attention_vectors[1], wgan.g.attention_vectors[2]])
    #Stitch together using attention
    #Make sure to rename objects that are the same name but different according to the attention
    #Return the argmaxed triples
    return np.argmax(gen_output, axis=2), disc_cost#np.mean(disc_cost, axis=1)

"""def loadAllValidationImages(path_to_val_batches):
    filenames = [os.path.join(path_to_val_batches, f) for f in os.listdir(path_to_val_batches) if f[-4:] == ".npz"]
    big_arr_list = []
    for f in range(len(filenames)):
        npz = np.load(filenames[f])
        big_arr_list.append(npz['arr_0'])
    return np.concatenate(big_arr_list, axis=0)"""

def drawSamples(wgan, session, filename, count, triples):
    samples_dir = wgan.samples_dir
    img = Image.open(filename)
    new_im = Image.new('RGB', (224 + 300, 224))
    img = img.resize((224, 224), Image.ANTIALIAS)
    text_im = Image.new('RGB', (300, 224))
    draw = ImageDraw.Draw(text_im)
    position = 0
    for i in xrange(min(19, len(triples))):
        s = " ".join(tuple(triples[i]))
        draw.text((10, 2 + (10*position)), s)
        position += 1
    new_im.paste(img, (0,0))
    new_im.paste(text_im, (224, 0))
    new_im.save(os.path.join(samples_dir, "{}.jpg".format(count)))

def decodeSamples(samples, decoder):
    all_samples = []

    for i in xrange(len(samples)):
        decoded = []
        for j in xrange(len(samples[i])):
            decoded.append(decoder[samples[i][j]])
        all_samples.append(decoded)

    return all_samples

#This will require that in the preprocessing phase, a small set of testing images
#is set aside. The extracted features will need to be mapped to the images they came
#from somehow. Need to figure out how to put real captions beside the generated ones. 
#Also potentially visualize attention here.
def generateSamples(params, session, wgan):
    path_to_val_batches = os.path.join(params["vg_batches"], "eval")
    #path_to_dict = os.path.join(path_to_val_batches, "filename_to_feats_dict.json")
    path_to_batch_0_filenames = os.path.join(path_to_val_batches, 'filenames.txt')

    #with open(path_to_dict, 'r') as f:
    #    filename_to_feats = json.load(f)
    with open(path_to_batch_0_filenames, 'r') as f:
        filenames = [line.strip() for line in f]

    samples_dir = wgan.samples_dir
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    batch = params["batch_size"]
    count = 0

    decoder = {y:x for x, y in vocab.iteritems()}

    for f in filenames:
        #im_feats = np.array([filename_to_feats[f]]*batch)
        #TODO Check in params if relations only is true
        samples, scores = generatePredictions(wgan, session)
        samples = decodeSamples(samples, decoder)
        drawSamples(wgan, session, f, count, samples)
        count += 1
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--vg_batches", default="./data/batches/", help="The path to the visual genome data. Defaults to ./data")
    parser.add_argument("--logs_dir", default="./logs/", help="The path to the logs where files will be saved and TensorBoard summaries are written.")
    parser.add_argument("--GPU", default="0", help="Which GPU to use")
    parser.add_argument("--samples_dir", default="./samples/", help="The path to the samples dir where samples will be generated.")
    parser.add_argument("--vocab", default="./preprocessing/saved_data/vocab.json", help="Path to the vocabulary")

    parser.add_argument("--batch_size", default=256, help="Batch size defaults to 256", type=int)
    parser.add_argument("--critic_iters", default=10, help="Number of iterations to train the critic", type=int)
    parser.add_argument("--generator", default="lstm", help="Generator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--discriminator", default="lstm", help="Discriminator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--epochs", default=30, help="Number of epochs defaults to 30", type=int)
    parser.add_argument("--print_interval", default=500, help="The model will be saved and samples will be generated every <print_interval> iterations", type=int)
    parser.add_argument("--tf_verbosity", default="ERROR", help="Sets tensorflow verbosity. Specifies which warning level to suppress. Defaults to ERROR")
    parser.add_argument("--lambda", default=10, help="Lambda term which regularizes to be close to one lipschitz", type=int)
    parser.add_argument("--use_language", default=False, help="Determines whether the generator update is also based on a discriminator trained on language only", type=bool)
    parser.add_argument("--dataset_relations_only", default=False, help="When true, indicates that the data only contains relations, and will affect how data is read", type=bool)


    args = parser.parse_args()
    params = vars(args)

    with open(params["vocab"], "r") as f:
        vocab = json.load(f)
        vocab_size = len(vocab)

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = params["GPU"]

    #Call this just in case the graph is persisting due to TF closing suddenly
    tf.reset_default_graph()

    with tf.Session() as sess:
        wgan = loadModel(params, sess)
        generateSamples(params, sess, wgan)
