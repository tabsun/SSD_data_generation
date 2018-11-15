import cv2
import os
import json
import requests
import urllib2
import hashlib
import socket
import xml.etree.ElementTree as ET 

def get_infos(fmid, anno):
    infos = []
    for wm_type, segments in anno.items():
        for segment in segments:
            start, end = segment[4:6]
            if(fmid > start and fmid < end):
                infos.append([wm_type]+segment[:4])
    return infos

def node_by_name_text(name, text):
    node = ET.Element(name)
    node.text = str(text)
    return node

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

root = "./generated_data/"
image_dir = "./generated_data/images"
anno_dir = "./generated_data/annotations"

colors = {"dy":(255,0,255), \
          "ks":(255,0,0), \
          "mp":(128,128,0), \
          "mp0":(0,255,0), \
          "mp1":(0,0,255), \
          "nh":(0,255,255), \
          "hs":(255,255,0), \
          "pp":(128,0,128) }

anames = []
for aname in os.listdir(anno_dir):
    if aname.endswith(".xml"):
        anames.append(os.path.join(anno_dir,aname))

count = 0
for aname in anames:
    print "%s %d / %d" % (aname, count, len(anames))

    tree = ET.parse(aname)
    root = tree.getroot()

    image_name = root.find('filename').text
    image = cv2.imread(os.path.join(image_dir, image_name))

    for obj in root.iter('object'):
        wm_type = obj.find('name').text
        color = colors[wm_type]
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, 2)
        cv2.putText(image, wm_type, (xmin,ymax), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)
    cv2.imshow("Check...", image)
    c = cv2.waitKey(0)
    if c == 27:
        break

    count += 1        

    
        
