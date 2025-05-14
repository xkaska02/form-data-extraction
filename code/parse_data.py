from pero_ocr.core.layout import PageLayout, LineString, RegionLayout, TextLine
import json
import numpy as np    
from shapely import Polygon
from utils import rectangle_from_ls, polygon_to_rect, draw_rect
import shutil


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

# a is json object with annotations for current file
def assign_labels_to_tokens(xml_file, a):
    page = PageLayout(file=xml_file)
    lines = page.lines_iterator()
    result = {
        "id":int,
        "tokens":[],
        "bboxes":[],
        "ner_tags":[],
        "image" : str
        }
    sorted_lines : list[TextLine] = []
    
    for l in lines:
        result["tokens"].append(l.transcription.split(' '))
        # print(l.polygon[0][0], l.polygon[0][1])
        # print(l.transcription)
        sorted_lines.append(l)
        
        
    # print(sorted_lines)
    idk : list[TextLine] = sorted(sorted_lines, key= lambda line: line.polygon[0][1])
    # for i in idk:
    #     print(i.transcription)
    #     print(i.polygon[0])

        
    sorted_labels = sorted(a["label"], key= lambda label: label['y'])
    for l in sorted_labels:
        convert_from_ls(l)
        # print(l['rectanglelabels'], l['x'], l['y'])
        
    
    # print(sorted_labels[0])
    # print(sorted_lines[0].polygon)
    
    # pero_polygon = Polygon(polygon_to_rect(sorted_lines[0].polygon))
    # label_polygon = Polygon(rectangle_from_ls(sorted_labels[0]))
    
    # pero_polygon = (polygon_to_rect(sorted_lines[0].polygon))
    # label_polygon = (rectangle_from_ls(sorted_labels[0]))
    
    # intersection = pero_polygon.intersection(label_polygon)
    # print(intersection.area)
    pero_polygons : list[Pero_polygon] = [] 
    label_polygons : list [Label_polygon] = []
    result_polygons : list [Result_polygon] = [] 
    
    for line in sorted_lines:
        # line.transcription # text
        # pero_polygons.append(Polygon(polygon_to_rect(line.polygon)))
        pero_polygons.append(Pero_polygon(Polygon(polygon_to_rect(line.polygon)),line.transcription))
        
    for label in sorted_labels:
        # label_polygons.append(Polygon(rectangle_from_ls(label)))
        label_polygons.append(Label_polygon(Polygon(rectangle_from_ls(label)),label['rectanglelabels'][0]))
        
    # 
    
    # for l in label_polygons:
    #     iou_max = 0 
    #     res_poly : Polygon = l
    #     for p in pero_polygons:
    #         intersection = p.intersection(l)
    #         union = p.union(l)
    #         iou = intersection.area / (union.area)
    #         if iou > iou_max:
    #             iou_max = iou
    #             res_poly = intersection
    #             print(iou)
    #     if res_poly:
    #         # print("res: ", res_poly.area)
    #         result_polygons.add(res_poly)
    
    # print(len(result_polygons))
    # here we go again other way around
    for p in pero_polygons:
        iou_max = 0
        res_poly : Result_polygon = Result_polygon(p.shape, 0.1, p.text, "None")
        for l in label_polygons:
            intersection = p.shape.intersection(l.shape)
            union = p.shape.union(l.shape)
            iou = intersection.area / (union.area)
            # print(f"union {union.area} intersection {intersection.area} iou {iou}\n")
            if iou > iou_max:
                iou_max = iou
                # res_poly = intersection
                res_poly.shape = intersection
                res_poly.iou = iou
                res_poly.category = l.category
                res_poly.text = p.text
                # print(iou)
        if res_poly.iou > 0.09:
            # print("res: ", res_poly.area)
            # print("IOUMAX",iou_max)
            result_polygons.append(res_poly)                    

    
    for r in result_polygons:
        print(vars(r))
    
    # for poly in result_polygons: 
    #     print(poly.area)
    # print(img_file)
    
    # tmp = 0
    #! drawing rectangles into images - currently commented for performance
    # out_file = 'data_files/forms/rects/' + img_file.split('/')[-1].split('.')[0] + ".png"
    # shutil.copy(img_file, out_file)
    
    # for p in pero_polygons:
    #     # draw_into_image(pero_rect=p.bounds,out_path="data_files/test.png",img="data_files/test.png")
    #     draw_rect(rect=p.shape.bounds, img=out_file, color="red")
    #     # tmp+=1
    #     # if tmp == 2:
    #     #     exit(0)
    #     print("pero polygon")
    
    # for p in label_polygons:
    #     draw_rect(rect=p.shape.bounds, img=out_file, color="blue")
    #     print("label polygon")
    #! end drawing rectangles    
    
    # uncomment these 2 lines if you have commented the drawing of rectangles above
    # out_file = 'data_files/forms/rects/' + img_file.split('/')[-1].split('.')[0] + "-intersection.png"
    # shutil.copy(img_file, out_file)
    # for r in result_polygons:
    #     if r.category == "None":
    #         draw_rect(rect=r.shape.bounds, img=out_file, color="red")
    #     else:            
    #         draw_rect(rect=r.shape.bounds, img=out_file, color="green")
    #     print(r.shape.bounds)
    
    ######################################################
    
    # for line in sorted_lines:
    #     # print(line.transcription)
    #     pero_polygon = Polygon(line.polygon)
    #     intersections = []
    #     for l in sorted_labels:
    #         label_polygon = Polygon(rectangle_from_ls(l))
    #         intersections.append(pero_polygon.intersection(label_polygon).area)
    #     # print(intersections)
    #     max_value = max(intersections)
    #     if max_value > 0:
    #         max_value_index = intersections.index(max_value)
    #         # print(sorted_labels[max_value_index]['rectanglelabels'])
    #         print(f"{line.transcription} : {sorted_labels[max_value_index]['rectanglelabels'][0]}")
            
        # draw_into_image(label_polygon.bounds, pero_polygon.bounds, intersections.bounds, img_file, "data_files/test.png")        
        
    
    
    
    

    
    
    # print(intersection)
    
    # this is not optimal at all, it is just trying every label for every line and choosing which one is closest
    # for i in idk:
    #     deviations = []
    #     for l in sorted_labels:
    #         # check for lowest deviation
    #         x_deviation = abs(i.polygon[0][0] - l['x'])
    #         y_deviation = abs(i.polygon[0][1] - l['y'])
    #         deviations.append(x_deviation+y_deviation)
        
        
    #     # 70 is a threshold because there are some lines without label and i just pick the closest label
    #     if deviations[np.argmin(deviations)] < 70:
    #         tokens_with_labels.append((i, sorted_labels[np.argmin(deviations)]))
    #     else:
    #         tokens_with_labels.append((i, []))
    # for t in tokens_with_labels:
    #     if t[1]:
    #         print(t[0].transcription, t[1]["rectanglelabels"])
    #     else:
    #         print(t[0].transcription, "[]")
    # results.append(result)
    # print(len(result['tokens']))
    


# convert from LS percent units to pixels 
def convert_from_ls(result):
    pixel_x = result['x'] / 100.0 * result['original_width']
    pixel_y = result['y'] / 100.0 * result['original_height']
    pixel_width = result['width'] / 100.0 * result['original_width']
    pixel_height = result['height'] / 100.0 * result['original_height']
    
    result['x'] = pixel_x
    result['y'] = pixel_y
    result['width'] = pixel_width
    result['height'] = pixel_height
    
with open("data_files/project-39-at-2024-12-03-12-38-18806692.json", 'r') as file:
    ## annotations from label studio
    annotations = json.load(file) 
    


cnt = 0

for a in annotations:
    img = str.split(a['image'],'/')[-1]
    img_file = "data_files/forms/images/" + img
    xml_file = "data_files/forms/pagexml/" + img.replace("jpg", "xml")
    print(f"---------------------{xml_file}---------------------")
    assign_labels_to_tokens(xml_file, a)
    
    print("============================================================")
    cnt+=1
    if cnt == 2:
        exit(0)
    
    