import cv2
import os
import json
import random
import requests
import hashlib
import socket
import xml.etree.ElementTree as ET 

multi_rate = 0.1
blur_possibility = 0.8
sample_num = 500
scales = [0.4,  0.6,  0.8, 1.0, 1.5,  2.0]
props  = [0.03, 0.07, 0.1, 0.6, 0.1, 0.1]

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

def is_overlap(boxes, box):
    t, b, l, r = box
    for _box in boxes:
        _t, _b, _l, _r = _box
        if(r > _l and l < _r and b > _t and t < _b):
            return True
    return False
    
def generate_image_with_wm(image, existed_infos, added_image, wm_type):
    seed = random.random() 
    scale_index = 0
    left_value = seed - props[scale_index]
    while(left_value > 0):
        scale_index += 1
        left_value -= props[scale_index]
    scale = scales[scale_index]

    if(wm_type == "pp"):
        ratio = 0.3 - random.random()*0.6
    else:
        ratio = 0.0
    
    # resize and ratio the added image
    h, w = added_image.shape[:2]
    h = int((1.0+ratio)*scale*h + 0.5)
    w = int(scale*w + 0.5)
    a_image = cv2.resize(added_image, (w,h))

    # put it on image in watermark manner
    boxes = [info[1:] for info in existed_infos]
    height, width = image.shape[:2]
    if not (width > w and height > h):
        return False, image, []
    px = int(random.random()*(width-w)+0.5)
    py = int(random.random()*(height-h)+0.5)
    while(is_overlap(boxes, [py, py+h, px, px+w])):
        px = int(random.random()*(width-w)+0.5)
        py = int(random.random()*(height-h)+0.5)

    if(wm_type == "pp"):
        mask = a_image[:,:,-1]
    else:
        mask = cv2.cvtColor(a_image, cv2.COLOR_BGRA2GRAY)
        ret, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        

    for y in range(h):
        for x in range(w):
            if(mask[y,x] > 50):
                if(wm_type == "nh"):
                    image[y+py,x+px,:] = a_image[y,x,:]
                else:
                    image[y+py,x+px,:] = a_image[y,x,:-1]

    # blurry operation
    if random.random() < blur_possibility:
        dx = cv2.convertScaleAbs(cv2.Sobel(mask, cv2.CV_16S, 0, 1))
        dy = cv2.convertScaleAbs(cv2.Sobel(mask, cv2.CV_16S, 1, 0))
        
        blur_mask = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
        blur_mask_bin = blur_mask.copy()
        cv2.normalize(blur_mask, blur_mask_bin, 0, 255, cv2.NORM_MINMAX)
        
        blur_image = cv2.GaussianBlur(image, (5,5), 0)
        for y in range(h):
            for x in range(w):
                if(blur_mask_bin[y,x] > 127):
                    image[y+py,x+px,:] = blur_image[y,x,:]
        
    return True, image, [wm_type, py, py+h, px, px+w]
    
if __name__ == '__main__':
    count = 0
    # Get all watermark transparent images
    wm_images = {}
    watermark_images_dir = "../added_watermarks"
    for fname in os.listdir(watermark_images_dir):
        if fname.endswith('.png'):
            wm_type = fname.split("_")[0]
            image = cv2.imread(os.path.join(watermark_images_dir, fname), cv2.IMREAD_UNCHANGED)
            if(wm_type in wm_images.keys()):
                wm_images[wm_type].append(image)
            else:
                wm_images[wm_type] = [image]
    # Get all neg images
    fpaths = []
    neg_images_dir = "../neg_images"
    for fname in os.listdir(neg_images_dir):
        if fname.endswith('.png'):
            fpaths.append(os.path.join(neg_images_dir, fname))

    accum = {}
    for wm_type in wm_images.keys():
        accum[wm_type] = 0

    for wm_type, added_images in wm_images.items():
        if(sample_num < len(fpaths)):
            sample_fpaths = random.sample(fpaths, sample_num)
        else:
            sample_fpaths = []
            while(len(sample_fpaths) < sample_num):
                if sample_num - len(sample_fpaths) < len(fpaths):
                    sample_fpaths += random.sample(fpaths, sample_num - len(sample_fpaths))
                else:
                    sample_fpaths += fpaths

        for fpath in sample_fpaths:
            infos = []
            image = cv2.imread(fpath)
            index = random.sample(range(len(added_images)), 1)[0]
            added_image = added_images[index]
            ret, image, info = generate_image_with_wm(image, infos, added_image, wm_type)
            if(not ret):
                continue

            infos.append(info)
            accum[wm_type] += 1
 
            if(random.random() < multi_rate):
                seed_wm_types = []
                for seed_wm_type in wm_images.keys():
                    if seed_wm_type != wm_type:
                        seed_wm_types.append(seed_wm_type)
                another_wm_type = random.sample(seed_wm_types, 1)[0]
                another_added_images = wm_images[another_wm_type]
                index = random.sample(range(len(another_added_images)),1)[0]
                ret, image, info = generate_image_with_wm(image, infos, another_added_images[index], another_wm_type)
                if(ret):
                    infos.append(info)
                    accum[another_wm_type] += 1
            
            image_path = os.path.join("./generated_images/images/%05d.png" % count)
            cv2.imwrite(image_path, image)
            anno_path = os.path.join("./generated_images/annotations/%05d.xml" % count)
            write_xml(anno_path, "%05d.png"%count, image, infos)
            count += 1
            print(count)
