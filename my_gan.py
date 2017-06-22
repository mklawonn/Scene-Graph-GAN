import os, sys
sys.path.append(os.getcwd())

import time
import json
import random
import argparse

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from subprocess import call

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


class SceneGraphWGAN(object):
    def __init__(self, batch_path, path_to_vocab_json, generator, discriminator, logs_dir, samples_dir, BATCH_SIZE=64, CRITIC_ITERS=10):
        self.batch_path = batch_path
        self.batch_path += "/" if self.batch_path[-1] != "/" else ""
        self.path_to_vocab_json = path_to_vocab_json
        self.path_to_vocab_json += "/" if self.path_to_vocab_json != "/" else ""
        self.configuration = "{}_gen_{}_disc_{}_critic".format(generator, discriminator, CRITIC_ITERS)
        self.logs_dir = os.path.join(logs_dir, self.configuration)
        self.checkpoints_dir = os.path.join(self.logs_dir, "checkpoints/")
        self.summaries_dir = os.path.join(self.logs_dir, "summaries/")
        self.samples_dir = os.path.join(samples_dir, self.configuration)
        self.initial_gumbel_temp = 2.0

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        else:
            print "WARNING: Checkpoints directory already exists for {} configuration. Files will be overwritten.".format(self.configuration)

        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)
        else:
            print "WARNING: Summaries directory already exists for {} configuration. Old files will be deleted.".format(self.configuration)

        if not os.path.exists(self.samples_dir):
            os.makedirs(self.samples_dir)
        else:
            print "WARNING: Samples directory already exists for {} configuration. Old files will be deleted".format(self.configuration)

        for f in os.listdir(self.summaries_dir):
            call(["rm", os.path.join(self.summaries_dir, f)])

        for f in os.listdir(self.samples_dir):
            call(["rm", "-rf", os.path.join(self.samples_dir, f)])

        #Calculating vocabulary and sequence lengths
        with open(path_to_vocab_json, "r") as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        self.decoder = {y[0]:x for x, y in self.vocab.iteritems()}
        self.seq_len = 3

        #Image feature dimensionality
        self.image_feat_dim = [196, 512]
        self.word_embedding_size = 300
        #self.image_feat_dim = 4096

        #Hyperparameters
        self.BATCH_SIZE = BATCH_SIZE
        self.LAMBDA = 10
        self.CRITIC_ITERS = CRITIC_ITERS
        self.DIM = 512
        self.ITERS = 100000


        #Import the correct discriminator according to the keyword argument
        if discriminator == "mlp":
            from architectures.mlp_discriminator import Discriminator
        elif discriminator == "conv1D":
            from architectures.conv1D_discriminator import Discriminator
        else:
            from architectures.discriminator_with_attention import Discriminator

        if generator == "mlp":
            from architecutres.mlp_generator import Generator
        elif generator == "conv1D":
            from architectures.conv1D_generator import Generator
        else:
            from architectures.generator_with_attention import Generator

        #Initialize all the generator and discriminator variables
        with tf.variable_scope("Generator") as scope:
            self.g = Generator(self.vocab_size, batch_size = self.BATCH_SIZE)

        with tf.variable_scope("Discriminator") as scope:
            self.d = Discriminator(self.vocab_size, batch_size = self.BATCH_SIZE)

    #def Generator(self, image_feats, noise, init_word_embedding, prev_outputs=None):
    def Generator(self, image_feats, batch_size, gumbel_temp, prev_outputs=None):
        print "Building Generator"
        with tf.variable_scope("Generator", reuse=True) as scope:
            generated_words = self.g.build_generator(image_feats, batch_size, gumbel_temp)
            return generated_words

    def Discriminator(self, triple_input, image_feats):
        print "Building Discriminator"
        with tf.variable_scope("Discriminator", reuse=True) as scope:
            logits = self.d.build_discriminator(image_feats, triple_input)
            return logits

    def Loss(self):
        #real_inputs_discrete = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE, self.seq_len])
        #real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
        self.real_inputs = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.vocab_size])
        self.image_feats = tf.placeholder(tf.float32, shape=[None, self.image_feat_dim[0], self.image_feat_dim[1]])
        self.batch_size_placeholder = tf.placeholder(tf.int32)
        self.gumbel_temp = tf.placeholder(tf.float32)
        #self.init_word_embedding = tf.placeholder(tf.float32, shape=[None, self.word_embedding_size])
        #self.noise_placeholder = tf.placeholder(tf.float32, shape=[None, self.image_feat_dim[1]])

        #fake_inputs = self.Generator(self.image_feats, self.noise_placeholder, self.init_word_embedding)
        fake_inputs = self.Generator(self.image_feats, self.batch_size_placeholder, self.gumbel_temp)
        #fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

        self.fake_inputs = fake_inputs

        disc_real = self.Discriminator(self.real_inputs, self.image_feats)
        disc_fake = self.Discriminator(fake_inputs, self.image_feats)

        #First get the average loss over each timestep
        disc_cost = tf.reduce_mean(disc_fake, axis=1) - tf.reduce_mean(disc_real, axis=1)
        gen_cost = -tf.reduce_mean(disc_fake, axis=1)

        #Then get the loss over the batch
        disc_cost = tf.reduce_mean(disc_cost)
        gen_cost = tf.reduce_mean(gen_cost)

        #Then get the loss over the batch
        #disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        #gen_cost = -tf.reduce_mean(disc_fake)

        tf.summary.scalar("Discriminator Cost", disc_cost)
        tf.summary.scalar("Generator Cost", gen_cost)

        # WGAN lipschitz-penalty
        alpha = tf.random_uniform(
            shape=[self.batch_size_placeholder,1,1], 
            minval=0.,
            maxval=1.
        )
        differences = fake_inputs - self.real_inputs
        interpolates = self.real_inputs + (alpha*differences)
        gradients = tf.gradients(self.Discriminator(interpolates, self.image_feats), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += self.LAMBDA*gradient_penalty

        self.disc_cost = disc_cost
        self.gen_cost = gen_cost

        train_variables = tf.trainable_variables()
        gen_params = [v for v in train_variables if v.name.startswith("Generator")]
        disc_params = [v for v in train_variables if v.name.startswith("Discriminator")]

        assert len(gen_params) > 0
        assert len(disc_params) > 0

        #optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)

        #gen_grads = optimizer.compute_gradients(gen_cost, var_list=gen_params)
        #disc_grads = optimizer.compute_gradients(disc_cost, var_list=disc_params)

        self.gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)
        #self.gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost, var_list=gen_params)
        #self.disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(disc_cost, var_list=disc_params)

        #self.gen_train_op = optimizer.apply_gradients(gen_grads)
        #self.disc_train_op = optimizer.apply_gradients(disc_grads)

        """for grad, var in gen_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradient", grad)

        for grad, var in disc_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradient", grad)"""

    def DataGenerator(self):
        train_path = os.path.join(self.batch_path, "train")
        filenames = [os.path.join(train_path, i) for i in os.listdir(train_path)]
        #Otherwise we do it in the same order every time
        random.shuffle(filenames)
        for f in filenames:
            npz = np.load(f)
            big_arr = npz['arr_0']
            all_pairs = []
            for i in xrange(0, big_arr.shape[0], 2):
                im_feats = big_arr[i]
                caps = big_arr[i+1]
                for c in xrange(caps.shape[0]):
                    all_pairs.append((im_feats, caps[c]))
            indices = list(range(len(all_pairs)))
            random.shuffle(indices)
            while len(indices) > 0:
                batch_size = self.BATCH_SIZE
                if len(indices) >= self.BATCH_SIZE:
                    im_batch = np.array([all_pairs[i][0] for i in indices[-self.BATCH_SIZE:]], dtype=np.float32)
                    triple_batch = np.array([all_pairs[i][1] for i in indices[-self.BATCH_SIZE:]])
                    t_batch = np.zeros((self.BATCH_SIZE, 3, self.vocab_size), dtype=np.float32)
                    for row in range(t_batch.shape[0]):
                        for token in range(t_batch.shape[1]):
                            t_batch[row, token, triple_batch[row, token]] = 1.0
                    del indices[-self.BATCH_SIZE:]
                else:
                    im_batch = np.array([all_pairs[i][0] for i in indices], dtype=np.float32)
                    triple_batch = np.array([all_pairs[i][1] for i in indices])
                    t_batch = np.zeros((len(indices), 3, self.vocab_size), dtype=np.float32)
                    for row in range(t_batch.shape[0]):
                        for token in range(t_batch.shape[1]):
                            t_batch[row, token, triple_batch[row, token]] = 1.0
                    batch_size = len(indices)
                    del indices[:]
                yield im_batch, t_batch, batch_size

    #This will require that in the preprocessing phase, a small set of testing images
    #is set aside. The extracted features will need to be mapped to the images they came
    #from somehow. Need to figure out how to put real captions beside the generated ones. 
    #Also potentially visualize attention here.
    def generateSamples(self, session, iteration, gumbel_temp):
        iteration = str(iteration)
        #For image in list in eval
        eval_image_path = os.path.join(self.batch_path, "eval")
        #Load the filename to feats dictionary
        path_to_dict = os.path.join(eval_image_path, "filename_to_feats_dict.json")
        with open(path_to_dict, 'r') as f:
            filename_to_feats = json.load(f)
        with open(os.path.join(eval_image_path, "filenames.txt"), 'r') as f:
            filenames = [line.strip() for line in f]
        #list_of_images = []
        count = 0
        samples_dir = os.path.join(self.samples_dir, iteration)
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
        for f in filenames:
            #Open the file
            #Load the corresponding features from the dictionary or pick from the input image batch
            batch = 19
            #noise = np.random.uniform(size=(batch, self.image_feat_dim[1]))
            #init_word_embedding = np.zeros((batch, self.word_embedding_size))
            im_feats = np.array([filename_to_feats[f]]*batch)
            img = Image.open(f)
            #Generate some amount of triples from the features
            samples = session.run(self.fake_inputs, feed_dict={self.image_feats : im_feats, \
                self.batch_size_placeholder : batch, self.gumbel_temp : gumbel_temp})
            samples = np.argmax(samples, axis=2)
            decoded_samples = []
            #new_im will contain the original image and the text
            new_im = Image.new('RGB', (224 + 300, 224))
            #Resize the original image
            img = img.resize((224, 224), Image.ANTIALIAS)
            #Create the image on which to write the text
            text_im = Image.new('RGB', (300, 224))
            #Write out the text
            draw = ImageDraw.Draw(text_im)
            position = 0
            for i in xrange(len(samples)):
                decoded = []
                for j in xrange(len(samples[i])):
                    decoded.append(self.decoder[samples[i][j]])
                s = " ".join(tuple(decoded))
                draw.text((10, 2 + (10*position)), s)
                position += 1
            #Paste the original image and text image together
            new_im.paste(img, (0,0))
            new_im.paste(text_im, (224, 0))
            new_im.save(os.path.join(samples_dir, "{}.jpg".format(count)))
            #Load the combined image as a numpy array
            #new_im.load()
            #new_im = np.array(new_im, dtype=np.float64)
            #Append the image to a list
            #list_of_images.append(new_im)
            count += 1
        #Stack the list into a tensor
        #tensorflow_images = tf.stack(list_of_images)
        #Write out the stacked tensor to an image summary
        #tf.summary.image("generated_triples", tensorflow_images, max_outputs = 4)
        
    def validate(self, session, gumbel_temp):
        eval_image_path = os.path.join(self.batch_path, "eval")
        #Pick a random batch of evaluation images
        random_batch_index = np.random.randint(0, 10)
        npz = np.load(os.path.join(eval_image_path, "batch_{}.npz".format(random_batch_index)))
        big_arr = npz['arr_0']
        all_pairs = []
        for i in xrange(0, big_arr.shape[0], 2):
            im_feats = big_arr[i]
            caps = big_arr[i+1]
            for c in xrange(caps.shape[0]):
                all_pairs.append((im_feats, caps[c]))
        indices = list(range(len(all_pairs)))
        random.shuffle(indices)
        num_losses = 0.0
        gen_total = 0.0
        disc_total = 0.0
        while len(indices) > self.BATCH_SIZE:
            im_batch = np.array([all_pairs[i][0] for i in indices[-self.BATCH_SIZE:]], dtype=np.float32)
            triple_batch = np.array([all_pairs[i][1] for i in indices[-self.BATCH_SIZE:]])
            t_batch = np.zeros((self.BATCH_SIZE, 3, self.vocab_size), dtype=np.float32)
            for row in range(t_batch.shape[0]):
                for token in range(t_batch.shape[1]):
                    t_batch[row, token, triple_batch[row, token]] = 1.0
            del indices[-self.BATCH_SIZE:]
            #Run the generator and discriminator loss on this batch
            num_losses += 1.0
            noise = np.random.uniform(size=(self.BATCH_SIZE, self.image_feat_dim[1]))
            init_word_embedding = np.zeros((self.BATCH_SIZE, self.word_embedding_size))
            feed_dict = {self.real_inputs:t_batch, self.image_feats:im_batch, \
                self.batch_size_placeholder : self.BATCH_SIZE, self.gumbel_temp : gumbel_temp}
            _gen_cost = session.run(self.gen_cost, feed_dict=feed_dict)
            _disc_cost = session.run(self.disc_cost, feed_dict=feed_dict)
            gen_total += _gen_cost 
            disc_total += _disc_cost
        gen_total = gen_total / num_losses
        disc_total = disc_total / num_losses
        return gen_total, disc_total

            

    def Train(self, epochs, print_interval):
        self.saver = tf.train.Saver()
        self.Loss()
        summary_op = tf.summary.merge_all()
        with tf.Session() as session:
            loss_print_interval = 100
            #self.generateSamples(session)
            writer = tf.summary.FileWriter(self.summaries_dir, session.graph)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()


            session.run(tf.global_variables_initializer())

            gen = self.DataGenerator()

            print "Training WGAN for {} epochs. To monitor training via TensorBoard, run python -m tensorflow.tensorboard --logdir {}"\
                .format(epochs, self.summaries_dir)

            start_time = time.time()
            gumbel_temp = self.initial_gumbel_temp
            max_iteration = 0
            for epoch in range(epochs):
                iteration = 0
                for im_batch, triple_batch, batch_size in self.DataGenerator():
                    if gumbel_temp <= 0.2:
                        gumbel_temp = gumbel_temp
                    else: 
                        gumbel_temp = self.initial_gumbel_temp - (((epoch*max_iteration)+iteration)*0.0001)

                    #noise = np.random.uniform(size=(batch_size, self.image_feat_dim[1]))
                    #init_word_embedding = np.zeros((batch_size, self.word_embedding_size))

                    #Make sure all the batch sizes are the same
                    #assert im_batch.shape[0] == triple_batch.shape[0] == noise.shape[0] == init_word_embedding.shape[0] == batch_size

                    #Train Generator
                    if iteration > 0:
                        #Track training statistics every ten iterations
                        _ = session.run(self.gen_train_op, feed_dict={self.image_feats:im_batch, self.real_inputs : triple_batch,\
                                            self.batch_size_placeholder : batch_size, self.gumbel_temp : gumbel_temp})
                        #_ = session.run(self.gen_train_op, feed_dict={self.image_feats:im_batch, self.batch_size_placeholder : batch_size}, options=run_options, run_metadata=run_metadata)
                        #writer.add_run_metadata(run_metadata, "Iteration %d generator" % iteration)
                        #_ = session.run(self.gen_train_op, feed_dict={self.image_feats:im_batch, self.noise:noise}, options=run_options, run_metadata=run_metadata)

                    #Train Critic
                    if iteration == 0:
                        #It takes quite a few iterations to train to optimality the first time
                        critic_iters = 25
                    else:
                        critic_iters = self.CRITIC_ITERS
                    for i in xrange(critic_iters):
                        #im_batch, triple_batch = gen.next()
                        _disc_cost, _ = session.run(
                            [self.disc_cost, self.disc_train_op],
                            feed_dict={self.real_inputs:triple_batch, self.image_feats:im_batch, self.batch_size_placeholder : batch_size, self.gumbel_temp : gumbel_temp}
                        )

                    if iteration % loss_print_interval == 0:
                        stop_time = time.time()
                        if iteration == 0:
                            duration = stop_time - start_time
                        else:
                            duration = (stop_time - start_time) / loss_print_interval
                        start_time = stop_time
                        summary, _gen_cost = session.run([summary_op, self.gen_cost], 
                            feed_dict={self.real_inputs:triple_batch, self.image_feats:im_batch, self.batch_size_placeholder : batch_size, self.gumbel_temp : gumbel_temp})
                        val_gen, val_disc = self.validate(session, gumbel_temp)
                        #_gen_cost = session.run(self.gen_cost, feed_dict={self.real_inputs:triple_batch, self.image_feats:im_batch, self.noise:noise})
                        writer.add_summary(summary, iteration)
                        writer.flush()

                        print "Time {}/itr, Step: {}, Gumbel: {}\n Training: generator loss: {}, discriminator loss: {}\n Eval: generator loss: {}, discriminator loss: {}".format(
                                duration, iteration, gumbel_temp, _gen_cost, _disc_cost, val_gen, val_disc)

                    if iteration % print_interval == 0:
                        self.generateSamples(session, ((epoch*max_iteration)+iteration), gumbel_temp)
                        self.saver.save(session, os.path.join(self.checkpoints_dir, "model.ckpt"), global_step=(epoch*max_iteration)+iteration)

                    iteration += 1
                max_iteration = iteration

def parseConfigFile(path_to_config_file = "./config.txt"):
    arg_dict = {}
    with open("./config.txt", "r") as f:
        for line in f:
            line_ = line.split()
            arg_dict[line_[0]] = line_[1]
    return arg_dict


if __name__ == "__main__":
    #"Permanent" arguments from the config file
    arg_dict = parseConfigFile()
    batch_path = os.path.join(arg_dict["visual_genome"], "batches")
    path_to_vocab_json = arg_dict["vocab"]
    logs_dir = arg_dict["logs"]
    samples_dir = arg_dict["samples"]


    #Argparse args
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=64, help="Batch size defaults to 64", type=int)
    parser.add_argument("--critic_iters", default=10, help="Number of iterations to train the critic", type=int)
    parser.add_argument("--generator", default="lstm", help="Generator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--discriminator", default="lstm", help="Discriminator defaults to LSTM with attention. See the architectures folder.")
    parser.add_argument("--epochs", default=30, help="Number of epochs defaults to 30", type=int)
    parser.add_argument("--resume", default=False, help="Resume training from the last checkpoint for this configuration", type=bool)
    parser.add_argument("--print_interval", default=500, help="The model will be saved and samples will be generated every <print_interval> iterations", type=int)
    parser.add_argument("--tf_verbosity", default="ERROR", help="Sets tensorflow verbosity. Specifies which warning level to suppress. Defaults to ERROR")

    args = parser.parse_args()
    params = vars(args)

    verbosity_dict = {"DEBUG" : 0, "INFO" : 1, "WARN" : 2, "ERROR" : 3}

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '{}'.format(verbosity_dict[params["tf_verbosity"]])

    #Begin training
    wgan = SceneGraphWGAN(batch_path, path_to_vocab_json, params["generator"], params["discriminator"], logs_dir, samples_dir, 
           BATCH_SIZE=params["batch_size"], CRITIC_ITERS=params["critic_iters"])
    wgan.Train(params["epochs"], params["print_interval"])
