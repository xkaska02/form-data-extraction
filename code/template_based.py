import json
from shapely import Polygon
from shapely import equals_exact

with open('data_files/forms_json_dataset/dataset_without_keys.json', 'r') as file:
    dataset = json.load(file)


label_list = ["B-name","B-rank","B-birth_date","B-nationality","B-death_date","B-funeral_date","B-grave_location","B-grave_id","B-information_source","B-death_book"]

# print(dataset[0]["bboxes"])

borders = [219, 358, 592, 727,944,1043,1240,1330,1535,1616]
keys = ["Příjmení a jméno", "Hodnost a pluk", "Datum a narození polit. okres, země", "Příslušnost (polit. okres, země)", "Datum a místo úmrtí (polit. okres, země)", "Datum pohřbu", "Označení hřbitova a místo (polit. okres, země)", "Oddělení, číslo hrobu", "Opsáno podle", "Úmrtní kniha: tom, fol.", "Tiskárna MNO. - 1163 - 36."]


# bounding boxes of regions
regions = []
for i in range(len(borders)):
    if i == 0:
        regions.append([0,0, 1240, borders[i]])
    else:
        regions.append([0, borders[i-1], 1240, borders[i]])
        

    

# bounding boxes of keys in 1 image in the same order as label_list
key_areas = [[[101,106,411,154]], [[103,254,385,300]], [[95,403,552,440],[94,442,361,475]],[[102,624,297,664],[97,670,356,693]],[[100,772,476,809],[97,816,362,843]],[[101,994,368,1032]],[[104,1067,576,1108],[101,1112,352,1139]],[[105,1292,481,1341]],[[106,1365,362,1411]],[[106,1577,523,1621]], [[101, 1658, 470,1688]]]

def rect_from_bbox(bbox):
    return [(bbox[0],bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0],bbox[3])]



# idk = Polygon(rect_from_bbox(key_areas[0]))

region_polygons = [Polygon(rect_from_bbox(p)) for p in regions]
# key_areas_polygons = []
# for i in range(len(key_areas)):
#     for j in range(len(key_areas[i])):
#         key_areas_polygons.append().append(rect_from_bbox(key_areas[i][j]))
# for key in key_areas:
#     for k in key:
#         print(k)
        
# exit(0)
key_areas_polygons = [[Polygon(rect_from_bbox(bbox)) for bbox in bbox_list] for bbox_list in key_areas]

# print(key_areas_polygons)
    
# exit(0)

# cycle based on y coordinate only

# for i in range(5):
#     print(dataset[i]["image_name"])
#     for word,bbox,ner_tag in zip(dataset[i]["words"],dataset[i]["bboxes"],dataset[i]["ner_tags"]):
#         # print(word,bbox,ner_tag)
#         for j in range(len(borders)):
#             if bbox[3] < borders[j] and not any(word.lower() in s.lower() for s in keys):
#                 print(word,label_list[j], bbox)
#                 break
# print(len(key_areas_polygons), len(region_polygons))
# exit(0)



intersection_notempty = False
for i in range(1):
    for word, bbox, ner_tag in zip(dataset[i]["words"],dataset[i]["bboxes"],dataset[i]["ner_tags"]):
        # print(word, bbox, ner_tag)
        word_poly = Polygon(rect_from_bbox(bbox))
        # check which region word belongs to first
        for j in range(len(region_polygons)):
            # print(region_polygons[j], word)
            # print(region_polygons[j].intersection(word_poly), word_poly)
            if equals_exact(region_polygons[j].intersection(word_poly), word_poly, 5, normalize=True):
                # print(f"{word} is in {label_list[j]} region")
                break 
        
        for k in range(len(key_areas_polygons[j])):
            # print("searching if word is matching key", word, word_poly, key_areas_polygons[j][k], label_list[j])
            intersection = word_poly.intersection(key_areas_polygons[j][k])
            # print(f"intersection of {word} {word_poly} with {key_areas_polygons[j][k]} is empty{intersection.is_empty} ")
            if not intersection.is_empty:
                intersection_notempty = True
                # print("matching position with key breaking", intersection, word_poly, word)
                break
            else:
                # print(f"no intersection", word, word_poly, key_areas_polygons[j][k])
                # print("NO INTERSECTION")
                continue 
        # print(f"end of search {word} {word_poly} intersection with {key_areas_polygons[j]} is {intersection_notempty}")
        if not intersection_notempty:
            print(f"{word} - {label_list[j]}")
        intersection_notempty = False
        # exit(0)
        # for j in range(len(key_areas_polygons)):
        #     # print(word_poly)
        #     # print(key_areas_polygons[j])
        #     intersection = word_poly.intersection(key_areas_polygons[j])
        #     if not intersection.is_empty:
        #         intersection_notempty = True
        #         break
        # if(intersection_notempty):
        #     print(f"{word} - key")
        # else:
        #     print(f"{word} {label_list[j]}")
        # intersection_notempty = False
                

            
        