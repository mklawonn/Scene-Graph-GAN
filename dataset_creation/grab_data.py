import os, sys
sys.path.append(os.getcwd())

import requests

def streamSaveLink(link, filename):
    r = requests.get(link, stream = True, verify=False)
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

def getVisualGenome(save_path):
    links = ["https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", "http://visualgenome.org/static/data/dataset/image_data.json.zip",
             "http://visualgenome.org/static/data/dataset/attributes.json.zip", "http://visualgenome.org/static/data/dataset/synsets.json.zip", "http://visualgenome.org/static/data/dataset/scene_graphs.json.zip"]
    for l in links:
        r = requests.get(l, stream = True, verify=False)
        filename = os.path.join(save_path, l.split("/")[-1])
        streamSaveLink(l, filename)
