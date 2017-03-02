import local as vg

if __name__ == "__main__":
    p = "/home/user/data/visual_genome/"
    p1 = "/home/user/data/visual_genome/by_id"
    vg.AddAttrsToSceneGraphs(dataDir=p)
    vg.SaveSceneGraphsById(dataDir=p, imageDataDir=p1)
    g = vg.GetSceneGraph(1000, images=p, imageDataDir=p1)
    print g.attributes
