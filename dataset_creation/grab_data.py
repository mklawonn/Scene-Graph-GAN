import os, sys
sys.path.append(os.getcwd())

import requests
import argparse

def streamSaveLink(link, filename):
    r = requests.get(link, stream = True)
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

def getVisualGenome(save_path):
    links = ["https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", "http://visualgenome.org/static/data/dataset/image_data.json.zip",
             "http://visualgenome.org/static/data/dataset/attributes.json.zip", "http://visualgenome.org/static/data/dataset/synsets.json.zip", "http://visualgenome.org/static/data/dataset/scene_graphs.json.zip"]
    for l in links:
        r = requests.get(l, stream = True)
        filename = os.path.join(save_path, l.split("/")[-1])
        streamSaveLink(l, filename)
    
def getVGGModel(save_path):
    link = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"
    r = requests.get(link, stream = True)
    #filename = "{}{}".format(save_path, link.split("/")[-1])
    filename = save_path
    streamSaveLink(link, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--visual_genome", default="./data/", help="The path to the visual genome data. Defaults to ./data")
    parser.add_argument("--logs_dir", default="./models/vgg/vgg16.tfmodel", help="The path to the logs where files will be saved and TensorBoard summaries are written.")

    args = parser.parse_args()
    params = vars(args)

    path_to_data = params["visual_genome"]
    path_to_model = params["vgg_tf_model"]

    #If the directory doesn't exist, create it
    if not os.path.exists(path_to_data):
        os.makedirs(path_to_data)
    #Get the visual genome data and save it there
    getVisualGenome(path_to_data)
    #Get the VGG model and save it there
    getVGGModel(path_to_model)
