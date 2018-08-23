# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages
import cv2
import argparse
import os
import json
from tqdm import tqdm
## Get Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", type=str,
	help="path to directory with videos")
ap.add_argument("-c", "--config", type=str,default="",
	help="path to input deep learning config")
ap.add_argument("-m", "--model", type=str,
	help="path to input deep learning model for embeddings")
ap.add_argument("-l","--layers",nargs='+',type=str,
	help="layer to extract")
ap.add_argument("-s","--frame_stride",type=int,default=10,help="Frame stride to skip redundant information")
ap.add_argument("-o","--output",type=str,help="Output JSON file",required=True)
args = vars(ap.parse_args())

# Directory
directory = args["dir"]	
filenames = [directory +"/" + f for f in os.listdir(directory)]

# frame stride
frame_stride = args["frame_stride"]

## Get Model
config = args["config"]
model =  args["model"]

layerNames = args["layers"]
net = cv2.dnn.readNet(config,model)

output = []
for _file in filenames[0:3]:
    print("Extracting from %s"%_file)
    vid = cv2.VideoCapture(_file)
    out_dict = {}
    if vid.isOpened():
        out_dict["filename"] = _file
        fps                  = vid.get(cv2.CAP_PROP_FPS)
        frames               = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        out_dict["fps"]      = fps
        out_dict["frames"]   = frames
        out_embeddings = []
        for i in tqdm(range(0,frames,frame_stride)):
            embedding = {}
            frame_id = i
            vid.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
            ret,frame = vid.read()
            if ret:
                embedding["frame_id"] = frame_id
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                    	(127.5, 127.5, 127.5), swapRB=True, crop=False)
                net.setInput(blob)
                net_output = net.forward(layerNames)
                for _id,out in enumerate(net_output):
	                embedding[layerNames[_id]] = out.flatten().tolist()
            if embedding.keys():
                out_embeddings.append(embedding)
                
        if len(out_embeddings)> 0:
            out_dict["embeddings"] = out_embeddings
        
    if out_dict.keys():
        output.append(out_dict)

with open(args["output"],"w") as fp:
    json.dump(output,fp)
