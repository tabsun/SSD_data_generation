import cv2
import os
import json
import requests
import urllib2
import hashlib
import socket
import xml.etree.ElementTree as ET 

def parse_xml(input_path):
    count = dict()
    tree = ET.parse(input_path)
    root = tree.getroot()
    for obj in root.iter('object'):
        tag = obj.find('name').text 
        if tag in count.keys():
            count[tag] += 1
        else:
            count[tag] = 1
    return count

def write_xml(output_path, image_name, image, infos):
    tree = ET.parse('sample.xml')
    root = tree.getroot()
    fnode = root.find('filename')
    fnode.text = image_name

    height, width = image.shape[:2]
    root.find('size').find('width').text = str(width)
    root.find('size').find('height').text = str(height)

    for info in infos:
        obj = ET.Element("object")
        obj.append(node_by_name_text("name",info[0]))
        obj.append(node_by_name_text("pose", "Unspecified"))
        obj.append(node_by_name_text("truncated",0))
        obj.append(node_by_name_text("difficult",0))
        bndbox = ET.Element("bndbox")
        ymin, ymax, xmin, xmax = info[1:]
        bndbox.append(node_by_name_text("xmin", xmin))
        bndbox.append(node_by_name_text("ymin", ymin))
        bndbox.append(node_by_name_text("xmax", xmax))
        bndbox.append(node_by_name_text("ymax", ymax))
        obj.append(bndbox)

        root.append(obj)

    tree.write(output_path)

anno_dir = "./annotations"
distr = dict()
for anno in os.listdir(anno_dir):
    input_path = os.path.join(anno_dir, anno)
    count = parse_xml(input_path)
    for key, num in count.items():
        if key in distr.keys():
            distr[key] += count[key]
        else:
            distr[key] = 0

for key, num in distr.items():
    print("%s : %d" % (key, num))
exit(0)

colors = {"dy":(255,0,255), \
          "ks":(255,0,0), \
          "mp0":(0,255,0), \
          "mp1":(0,0,255), \
          "nh":(0,255,255), \
          "hs":(255,255,0), \
          "pp":(128,0,128) }

processed_tags = []
for aname in os.listdir(anno_dir):
    processed_tags.append(aname.split("_")[0])
if os.path.exists("processed_tags.txt"):
    with open("processed_tags.txt", "r") as f:
        for line in f.readlines():
            processed_tags.append(line.strip())

vnames = []
for vname in os.listdir(root):
    if vname.endswith(".mp4"):
        tag = vname.split(".mp4")[0]
        if(tag in processed_tags):
            continue
        vnames.append(os.path.join(root,vname))

count = 0
for vname in vnames:
    print "%s %d / %d" % (vname, count, len(vnames))
    tag = vname.split(".mp4")[0]
    with open("processed_tags.txt","a+") as f:
        f.write("%s\n" % tag)

    anno = vname.replace(".mp4",".json")
    with open(anno, "r") as f:
        annotation = dict(json.loads(f.readlines()[0].strip()))

    cap = cv2.VideoCapture(vname)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_time_step = max(int(fps*2), max(1, int(frame_num / 20+0.5)))

    frame_id = sample_time_step//2
    while(frame_id < frame_num):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        frame_id += sample_time_step
        if(not ret):
            continue

        infos = get_infos(frame_id, annotation)
        if(len(infos) == 0):
            continue
        
        left_infos = []
        for info in infos:
            color = colors[info[0]]
            t,b,l,r = info[1:]
            image = frame.copy()
            h, w = image.shape[:2]
            scale = 960.0 / max(h,w)
            nh, nw = int(scale*h), int(scale*w)
            image = cv2.resize(image, (nw,nh))
            t,b,l,r = map(lambda x: int(x*scale), [t,b,l,r])
            cv2.rectangle(image, (l,t), (r,b), color, 2)
            cv2.putText(image, info[0], (l,b), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)
            cv2.imshow("Check...", image)
            c = cv2.waitKey(0)

            if c == 27:
                exit(0)
            if c != 100:
                left_infos.append(info)
            else:
                continue

        if(len(left_infos) != 0):
            image_name = "%s_%05d.png"%(vname.split(".mp4")[0].split("/")[-1], frame_id)
            cv2.imwrite(os.path.join(image_dir,image_name), frame)
            output_path = os.path.join(anno_dir, image_name.replace(".png",".xml"))
            write_xml(output_path, image_name, frame, left_infos)

        
    cap.release()
    count += 1        

    
        
