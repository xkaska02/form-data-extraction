from pero_ocr.core.layout import PageLayout, LineString, RegionLayout, TextLine
import xml.etree.ElementTree as ET
from utils import draw_rect

def get_coords(HPOS: int, VPOS: int, HEIGHT: int, WIDTH: int):
    # return [(HPOS, VPOS), (HPOS, VPOS + HEIGHT), (HPOS + WIDTH, VPOS + HEIGHT), (HPOS + WIDTH, VPOS)]
    return [(HPOS, VPOS), (HPOS + WIDTH, VPOS + HEIGHT)]
    

# tree = ET.parse("data_files/forms/alto/00078bed-8658-47a4-96c6-c737824e9517.xml")
tree = ET.parse("data_files/forms/alto/0a1cf4b0-f841-40de-82b7-1619cadec7c5.xml")
root = tree.getroot()

layout = root[1]
page = layout[0]
printspace = page[4]


for textblock in printspace:
    for textline in textblock:
        for word in textline:
            if "String" in word.tag:
                rect = get_coords(int(word.attrib["HPOS"]),int(word.attrib["VPOS"]),int(word.attrib["HEIGHT"]),int(word.attrib["WIDTH"]))
                draw_rect(rect=rect, img="data_files/forms/rects/0a1cf4b0-f841-40de-82b7-1619cadec7c5.png", color="red")
        
        # for elem in textline:
        #     # print(elem.attrib)
        #     rect = get_coords(int(elem.attrib["HPOS"]),int(elem.attrib["VPOS"]),int(elem.attrib["HEIGHT"]),int(elem.attrib["WIDTH"]))
        #     #! for now hard code the picture to draw into
        #     draw_rect(rect=rect, img="data_files/forms/rects/0a1cf4b0-f841-40de-82b7-1619cadec7c5.png", color="blue")
        #     print(elem.attrib["CONTENT"])
        #     # print(elem.attrib)



# HPOS VPOS levy horni roh 
# HPOS VPOS + WIDTH
# HPOS + HEIGHT VPOS + WIDTH
# HPOS + HEIGHT VPOS

# page = PageLayout()
# page.from_altoxml(file="data_files/forms/alto/00078bed-8658-47a4-96c6-c737824e9517.xml")



# lines = page.lines_iterator()

# # print(next(lines).transcription)

# # for l in lines:
# #     print(l.transcription)

# for r in page.regions:
#     # print(vars(r))
#     for line in r.lines:
#         print(vars(line))