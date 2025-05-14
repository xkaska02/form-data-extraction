import xml.etree.ElementTree as ET
import json
from utils import convert_from_ls, rectangle_from_ls, rect_from_alto, polygon_to_rect
from shapely import Polygon
import sys

"""Class for objects from peroOCR to store bbox and text"""
class Pero_polygon:
    def __init__(self, shape : Polygon, text : str):
        self.shape = shape
        self.text = text
        
"""Class for objects from LabelStudio to store bbox and category"""
class Label_polygon:
    def __init__(self, shape : Polygon, category : str):
        self.shape = shape
        self.category = category

"""Class for merged result object from intersection over union"""
class Result_polygon:
    def __init__(self, shape : Polygon, iou : float, text : str, category : str):
        self.shape = shape
        self.iou = iou
        self.text = text
        self.category = category
        self.bbox = []

with open("data_files/project-39-at-2024-12-03-12-38-18806692.json", 'r') as file:
    ## annotations from label studio
    annotations = json.load(file) 


a_id : int = 0
"""annotation id"""

results = []

for a in annotations:
    result_json = {
        "id" : a_id,
        "words" : [],
        "bboxes" : [],
        "ner_tags" : [],
        "image_name" : ""
    }
    a_id += 1
    img = str.split(a['image'],'/')[-1]
    img_file = "data_files/forms/images/" + img
    xml_file = "data_files/forms/alto/" + img.replace("jpg", "xml")
    
    result_json["image_name"] = img_file
    
    for label in a['label']:
        if "closed" in label:
            for point in label['points']:
                point[0] = round(point[0]/100*label['original_width'])
                point[1] = round(point[1]/100*label['original_height'])
        else:
            convert_from_ls(label)
    try:
        tree = ET.parse(xml_file)
    except:
        sys.stderr.write(f"File {xml_file} not found skipping")
        continue
    
    words = []
    
    root = tree.getroot()

    layout = root[1]
    page = layout[0]
    printspace = page[4]
    for textblock in printspace:
        for textline in textblock:
            for text in textline:
                if "String" in text.tag:
                    words.append(text)
    sorted_words = sorted(words, key=lambda word: (int(word.attrib['VPOS'])//30, int(word.attrib['HPOS'])))
            
    pero_polygons = []
    label_polygons = []
    result_polygons = []
    
    for label in a['label']:
        label_polygons.append(Label_polygon(Polygon(rectangle_from_ls(label)), label['rectanglelabels'][0]))
    
    
    for word in sorted_words:
        pero_polygons.append(Pero_polygon(Polygon(polygon_to_rect(rect_from_alto(word))),word.attrib['CONTENT']))
    
    for p in pero_polygons:
        iou_max = 0
        res_poly : Result_polygon = Result_polygon(p.shape, 0.1, p.text, "None")
        for l in label_polygons:
            intersection = p.shape.intersection(l.shape)
            union = p.shape.union(l.shape)
            iou = intersection.area / (union.area)
            if iou > iou_max:
                iou_max = iou
                res_poly.shape = intersection
                res_poly.iou = iou
                res_poly.category = l.category
                res_poly.text = p.text
        if res_poly.iou > 0.09:
            result_polygons.append(res_poly)                    

    
    for r in result_polygons:
        min_x,min_y,max_x,max_y = r.shape.bounds
        r.bbox = [min_x, min_y, max_x, max_y]
        r.bbox = [int(i) for i in r.bbox] # converting to ints
        
        
        # creating json result that i can export 
        result_json["bboxes"].append(r.bbox)
        result_json["words"].append(r.text)
        label2id = {
            "None":0,
            "key":1,
            "information":2,
            "name":3,
            "rank":4,
            "birth_date":5,
            "nationality": 6,
            "death_date": 7,
            "funeral_date":8,
            "grave_location":9,
            "grave_id":10,
            "information_source":11,
            "death_book":12,
        }
        result_json["ner_tags"].append(label2id[r.category])

    results.append(result_json)
    
    json_object = json.dumps(results)
    
    with open("data_files/forms_json_dataset/dataset_without_keys.json","w") as file:
        file.write(json_object)

    