from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


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

    score_threshold = determineScoreThreshold(wgan, session)
    iou_threshold = 0.8

    wgan.custom_runner.start_threads(session)

    for f in filenames:
        #im_feats = np.array([filename_to_feats[f]]*batch)
        #TODO Check in params if relations only is true
        #samples, scores = generatePredictions(wgan, session)
        samples = generateAndFilterPredictions(wgan, session, iou_threshold, score_threshold)
        samples = decodeSamples(samples, decoder)
        drawSamples(wgan, session, f, count, samples)
        count += 1
 

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

#TODO Nicer graph drawing
