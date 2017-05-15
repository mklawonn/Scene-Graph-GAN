from subprocess import call

def unzipAll(save_path):
    for f in os.listdir(save_path):
        if f[-4:] == ".zip":
            call(["unzip", f])

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
