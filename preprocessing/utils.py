import skimage
import skimage.io
import skimage.transform
import numpy as np

from os.path import expanduser
home = expanduser("~")
if not home.endswith("/"):
    home = home + "/"

#synset = [l.strip() for l in open('synset.txt').readlines()]

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
  # load image
  img = skimage.io.imread(path)
  img = img / 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  #print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  return resized_img

# returns the top1 string
def print_prob(prob):
  #print prob
  print "prob shape", prob.shape
  pred = np.argsort(prob)[::-1]

  # Get top1 label
  top1 = pred[0]
  #top1 = synset[pred[0]]
  print "Top1: ", top1
  # Get top5 label
  top5 = [pred[i] for i in range(5)]
  #top5 = [synset[pred[i]] for i in range(5)]
  print "Top5: ", top5
  return top1

def compute_mean_for_images(image_dir):
    num_images = 0
    r_total = g_total = b_total = 0.0
    r_mean = g_mean = b_mean = 0.0
    for image_file in os.listdir(image_dir):
        img = cv2.imread(image_dir + image_file).astype(np.float32)
        r_total += np.sum(img[:,:,0]) / (img[:,:,0].shape[0] * img[:,:,0].shape[1])
        g_total += np.sum(img[:,:,1]) / (img[:,:,1].shape[0] * img[:,:,1].shape[1])
        b_total += np.sum(img[:,:,2]) / (img[:,:,2].shape[0] * img[:,:,2].shape[1])
        num_images += 1

    r_mean = r_total / float(num_images)
    g_mean = g_total / float(num_images)
    b_mean = b_total / float(num_images)

    return r_mean, g_mean, b_mean
