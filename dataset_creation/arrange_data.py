from subprocess import call
import os

def unzipAll(save_path):
    for f in os.listdir(save_path):
        if f[-4:] == ".zip":
            call(["unzip", f])

def addAttributes(save_path):
    attr_data = json.load(open(os.path.join(save_path, 'attributes.json')))
    with open(os.path.join(dataDir, 'scene_graphs.json')) as f:
        sg_dict = {sg['image_id']:sg for sg in json.load(f)}

    id_count = 0
    for img_attrs in attr_data:
        attrs = []
        for attribute in img_attrs['attributes']:
            a = img_attrs.copy(); del a['attributes']
            a['attribute']    = attribute
            a['attribute_id'] = id_count
            attrs.append(a)
        id_count += 1
        iid = img_attrs['image_id']
        sg_dict[iid]['attributes'] = attrs

    with open(os.path.join(save_path, 'scene_graphs.json'), 'w') as f:
        json.dump(sg_dict.values(), f)
    del attr_data, sg_dict
    gc.collect()


if __name__ == "__main__":
    #Read in path to the data
    with open("../config.txt", "r") as f:
        for line in f:
            line_ = line.split()
            if line_[0] == "visual_genome":
                path_to_data = line_[1]
    path_to_data += "/" if path_to_data[-1] != "/" else ""
    #Unzip everything
    unzipAll(path_to_data)
    #Create an "all_images" directory
    all_images = "{}all_images".format(path_to_data)
    if not os.path.exists(all_images):
        os.makedirs(all_images)
    #Move all images to "all_images"
    call(["mv", "{}images/*".format(path_to_data), all_images])
    call(["mv", "{}images2/*".format(path_to_data), all_images])
    #Add attributes to scene graphs
    addAttributes(path_to_data)
