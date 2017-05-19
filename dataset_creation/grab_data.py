import requests
import os

def streamSaveLink(link, filename):
    r = requests.get(link, stream = True)
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

#TODO Figure out what's actually necessary to download
def getVisualGenome(save_path):
    links = ["https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", "http://visualgenome.org/static/data/dataset/image_data.json.zip",
             "http://visualgenome.org/static/data/dataset/attributes.json.zip", "http://visualgenome.org/static/data/dataset/synsets.json.zip", "http://visualgenome.org/static/data/dataset/scene_graphs.json.zip"]
    for l in links:
        r = requests.get(l, stream = True)
        filename = "{}{}".format(save_path, l.split("/")[-1])
        streamSaveLink(l, filename)
    
#TODO Make available via a google drive link or something
def getVGGModel(save_path):
    link = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"
    r = requests.get(link, stream = True)
    #filename = "{}{}".format(save_path, link.split("/")[-1])
    filename = save_path
    streamSaveLink(link, filename)

if __name__ == "__main__":
    #Read in visual genome parameter from ../config.txt
    with open("../config.txt", "r") as f:
        for line in f:
            line_ = line.split()
            if line_[0] == "visual_genome":
                path_to_data = line_[1]
            elif line_[0] == "vgg_tf_model":
                path_to_model = line_[1]
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    #path_to_model += "/" if path_to_model[-1] != "/" else ""
    #If the directory doesn't exist, create it
    if not os.path.exists(path_to_data):
        os.makedirs(path_to_data)
    #Get the visual genome data and save it there
    getVisualGenome(path_to_data)
    #Get the VGG model and save it there
    getVGGModel(path_to_model)
