from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import generate_samples
from generate_samples import generatePredictions, init
from construct_graph import *
from train import SceneGraphWGAN

import argparse
import time


def drawSamples(wgan, session, filename, count, triples):
    samples_dir = wgan.samples_dir
    img = Image.open(filename)
    new_im = Image.new('RGB', (224 + 300, 224))
    img = img.resize((224, 224), Image.ANTIALIAS)
    text_im = Image.new('RGB', (300, 224))
    draw = ImageDraw.Draw(text_im)
    position = 0
    for i in xrange(min(19, len(triples))):
        #s = " ".join(tuple(triples[i]))
        s = triples[i]
        draw.text((10, 2 + (10*position)), s)
        position += 1
    new_im.paste(img, (0,0))
    new_im.paste(text_im, (224, 0))
    new_im.save(os.path.join(samples_dir, "{}.jpg".format(count)))

def loadModel(params, sess):
    #Create WGAN instance
    wgan = SceneGraphWGAN(params["vg_batches"], params["vocab"], params["generator"], params["discriminator"], params["logs_dir"], params["samples_dir"], 
           BATCH_SIZE=params["batch_size"], CRITIC_ITERS=params["critic_iters"], LAMBDA=params["lambda"], im_and_lang=params["use_language"],
           validation = True, resume=False, dataset_relations_only = params["dataset_relations_only"])
    wgan.constructOps()
    wgan.loadModel(sess)
    init()
    generate_samples.decoder = {y:x for x,y in wgan.vocab.iteritems()}
    queue_var_name = wgan.queue_var_name
    #return tf.get_default_graph().as_graph_def()
    tf.train.start_queue_runners(sess=sess)
    wgan.custom_runner.start_threads(sess) 
    #wgan.custom_runner.start_threads(sess)
    return wgan


#TODO Nicer graph drawing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vg_batches", default="./data/batches/", help="The path to the visual genome data. Defaults to ./data")
    parser.add_argument("--vg_images", default = "./data/all_images/", help="The path to the actual vg images")
    parser.add_argument("--logs_dir", default="./logs/", help="The path to the logs where files will be saved and TensorBoard summaries are written.")
    parser.add_argument("--GPU", default="0", help="Which GPU to use")
    parser.add_argument("--samples_dir", default="./samples/", help="The path to the samples dir where samples will be generated.")
    parser.add_argument("--vocab", default="./preprocessing/saved_data/vocab.json", help="Path to the vocabulary")
    parser.add_argument("--critic_iters", default=10)

    parser.add_argument("--batch_size", default=256, help="Batch size defaults to 256", type=int)
    parser.add_argument("--generator", default="lstm", help="Generator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--discriminator", default="lstm", help="Discriminator defaults to LSTM with attention. See the architectures folder.")
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
        print "Done loading, sleeping for some amount of time to allow the queue to populate"
        time.sleep(20)
        count = 0
        eval_path = os.path.join(params["vg_batches"], "eval")
        with open(os.path.join(eval_path, "filenames.txt"), "r") as f:
            filenames = [i.strip() for i in f]
        for filename in filenames:
            triples = generatePredictions(wgan, sess)
            all_entities = findAllEntities(triples)
            potential_duplicates = determinePotentialDuplicates(list(all_entities))
            resolveDuplicateEntities(potential_duplicates, all_entities, triples)
            #Sort triples
            sorted_triples = triples[:]
            sorted_triples.sort(key=lambda x : x.disc_score, reverse=True)
            sorted_triples = sorted_triples[:15]


            printable_triples = [i.decode() for i in sorted_triples]
            drawSamples(wgan, sess, filename, count, printable_triples)
            #TODO Rewrite recall at k and measure performance functions to compare the following two
            #ground truth things to the triples. By this line, triples should be comparable (e.g resolved)
            #Will need to account for the appended id though (e.g person3 should just be person)
            #gt_rels = wgan.custom_runner.gt_rels.get()
            #gt_atts = wgan.custom_runner.gt_atts.get()
            count += 1
