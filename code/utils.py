"""utility functions
"""

from PIL import Image, ImageDraw

# returns list of points to form a rectangle
def rectangle_from_ls(label):
    if "closed" in label:
        
        min_x = min(label['points'])[0]
        max_x = max(label['points'])[0]
        min_y = min(label['points'], key=lambda p: p[1])[1]
        max_y = max(label['points'], key=lambda p: p[1])[1]
        
        return [(min_x,min_y),(max_x, min_y),(max_x,max_y),(min_x, max_y)]
        
    return [(label['x'],label['y']),(label['x']+label['width'],label['y']),(label['x']+label['width'],label['y']+label['height']),(label['x'],label['y']+label['height'])]
    
# convert from LS percent units to pixels 
def convert_from_ls(result):
    pixel_x = result['x'] / 100.0 * result['original_width']
    pixel_y = result['y'] / 100.0 * result['original_height']
    pixel_width = result['width'] / 100.0 * result['original_width']
    pixel_height = result['height'] / 100.0 * result['original_height']
    
    result['x'] = round(pixel_x)
    result['y'] = round(pixel_y)
    result['width'] = round(pixel_width)
    result['height'] = round(pixel_height)
    
def polygon_to_rect(p):
    x_min = min(p)[0]
    y_min = min(p, key=lambda t: t[1])[1]
    x_max = max(p)[0]
    y_max = max(p, key=lambda t: t[1])[1]
    
    return [(x_min, y_min),(x_max, y_min),(x_max, y_max),(x_min,y_max)]

def rect_from_alto(elem):
    x_min = int(elem.attrib['HPOS'])
    y_min = int(elem.attrib['VPOS'])
    x_max = x_min + int(elem.attrib['WIDTH'])
    y_max = y_min + int(elem.attrib['HEIGHT'])
    
    return [(x_min, y_min),(x_max, y_min),(x_max, y_max),(x_min,y_max)]
    

    """draws a rectangle in the image supplied
    """
def draw_rect(rect, img, color):
    # img = 'data_files/forms/images/00078bed-8658-47a4-96c6-c737824e9517.jpg'
    out_file = 'data_files/forms/rects/' + img.split('/')[-1].split('.')[0] + ".png"
    # print(out_file)
    if img != None:
        with Image.open(img) as im:
            draw = ImageDraw.Draw(im)
            draw.rectangle(rect, outline=color)
            im.save(out_file, "PNG")
    

def draw_into_image(label_rect=None, pero_rect=None, intersect=None, img=None, out_path=None):
    if img != None:
        with Image.open(img) as im:
            draw = ImageDraw.Draw(im)
            if pero_rect != None:
                draw.rectangle(pero_rect, outline='red')
            if label_rect != None:
                draw.rectangle(label_rect,outline='blue')
            if intersect != None:
                draw.rectangle(intersect,outline='violet')
            if out_path != None:
                im.save(out_path,"PNG")
                
def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height)
    ]
    
def generate_model_name(base="lilt_robeczech", lr=5e-5, batch_size=8, train_size=30, extra=""):
  if extra != "":
    return f"{base}_lr{lr}_bs{batch_size}_train{train_size}_{extra}"
  else:
    return f"{base}_lr{lr}_bs{batch_size}_train{train_size}"