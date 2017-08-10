import tensorflow as tf
import numpy as np

def loadModel(params, sess):
    #Create WGAN instance
    wgan = SceneGraphWGAN(params["visual_genome"], params["vocab"], params["generator"], params["discriminator"], params["logs_dir"], params["samples_dir"], 
           BATCH_SIZE=params["batch_size"], CRITIC_ITERS=params["critic_iters"], LAMBDA=params["lambda"], im_and_lang=params["use_language"], resume=False)
    wgan.constructOps()
    wgan.loadModel(sess)
    return tf.get_default_graph().as_graph_def()

def generatePredictions(image_features, graph_def, sess, relations_only = True):
    #Enqueue image features
    #Initialize the queue variables
    dummy_triples = np.zeros((image_features.shape[0], 3, vocab_size), dtype=np.float32)
    if relations_only:
        feed_dict = {im_feats_placeholder : image_features, triples_placeholder : dummy_triples, flag_placeholder : relations_flag}
        #Generate a whole bunch of triples
        #TODO: If the Add_14 tensor gets renamed make sure to change it here
        triples_tensor = graph_def.get_tensor_by_name("Generator_1/generator_output:0")
        scores_tensor = graph_def.get_tensor_by_name("Discriminator_2/Add_14:0")
        
        sub_att_vector = graph_def.get_tensor_by_name("Generator_1/attention_softmax:0")
        pred_att_vector = graph_def.get_tensor_by_name("Generator_1/attention_softmax_1:0")
        obj_att_vector = graph_def.get_tensor_by_name("Generator_1/attention_softmax_2:0")

        triples, scores, sub_att, pred_att, obj_att = sess.run([triples_tensor, scores_tensor, sub_att_vector, pred_att_vector, obj_att_vector], feed_dict=feed_dict)
    else:
        pass
    #Stitch together using attention
    #Make sure to rename objects that are the same name but different according to the attention
    #Return the argmaxed triples
    return np.argmax(triples, axis=2), np.mean(scores, axis=1)

def loadAllValidationImages(path_to_val_batches):
    filenames = [os.path.join(path_to_val_batches, f) for f in os.listdir(path_to_val_batches) if f[-4:] == ".npz"]
    big_arr_list = []
    for f in range(len(filenames)):
        npz = np.load(filenames[f])
        big_arr_list.append(npz['arr_0'])
    return np.concatenate(big_arr_list, axis=0)

def drawSamples(params, session, filename, count, triples):
    samples_dir = params["samples_dir"]
    img = Image.open(filename)
    new_im = Image.new('RGB', (224 + 300, 224))
    img = img.resize((224, 224), Image.ANTIALIAS)
    text_im = Image.new('RGB', (300, 224))
    draw = ImageDraw.Draw(text_im)
    position = 0
    for i in xrange(len(triples)):
        s = " ".join(tuple(triples))
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
def generateSamples(params, session, graph_def):

    path_to_val_batches = params["visual_genome"]
    path_to_dict = os.path.join(path_to_val_batches, "filename_to_feats_dict.json")
    path_to_batch_0_filenames = os.path.join(path_to_val_batches, 'filenames.txt')

    with open(path_to_dict, 'r') as f:
        filename_to_feats = json.load(f)
    with open(path_to_batch_0_filenames, 'r') as f:
        filenames = [line.strip() for line in f]
    with open(path_to_vocab, 'r') as f:
        vocab = json.load(f)

    samples_dir = os.path.join(samples_dir, iteration)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    batch = 19
    count = 0

    decoder = {y[1]:x for x, y in vocab.iteritems()}

    for f in filenames:
        im_feats = np.array([filename_to_feats[f]]*batch)
        samples = generatePredictions(im_feats, graph_def, session, relations_only = True)
        samples = decodeSamples(samples, decoder)
        drawSamples(params, session, f, count, triples)
        count += 1
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--visual_genome", default="./data/batches/", help="The path to the visual genome data. Defaults to ./data")
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
        model_def = loadModel(sess, params)
        generateSamples(params, session, graph_def):
        
