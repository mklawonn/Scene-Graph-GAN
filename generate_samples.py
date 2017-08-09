import tensorflow as tf
import numpy as np


#This will require that in the preprocessing phase, a small set of testing images
#is set aside. The extracted features will need to be mapped to the images they came
#from somehow. Need to figure out how to put real captions beside the generated ones. 
#Also potentially visualize attention here.
def generateSamples(session, iteration, batch_path, samples_dir):
    iteration = str(iteration)
    #For image in list in eval
    eval_image_path = os.path.join(batch_path, "eval")
    #Load the filename to feats dictionary
    path_to_dict = os.path.join(eval_image_path, "filename_to_feats_dict.json")
    with open(path_to_dict, 'r') as f:
        filename_to_feats = json.load(f)
    with open(os.path.join(eval_image_path, "filenames.txt"), 'r') as f:
        filenames = [line.strip() for line in f]
    #list_of_images = []
    count = 0
    samples_dir = os.path.join(samples_dir, iteration)
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
        samples = session.run(self.fake_inputs, feed_dict={self.image_feats : im_feats,\
            self.batch_size_placeholder : batch, self.attribute_or_relation : float(count%2), self.gumbel_temp : gumbel_temp})
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
