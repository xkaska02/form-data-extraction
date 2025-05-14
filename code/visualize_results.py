from PIL import Image, ImageDraw, ImageFont
from utils import draw_rect
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file", default=None, help="file with the outputs of model")
    parser.add_argument("--save_images", default=True, type=lambda x: x.lower() == "true", help="if true images will be saved else images will only be shown")
    parser.add_argument("--save_path", default=None, help="path to save images to if `save_images` is set, default is `input_file`")
    
    args = parser.parse_args()
    return args

def main(args):

    input_file = args.input_file


    with open(input_file, "r") as f:
        results = json.load(f)


    label2color_fill = {
                "B-key":(255,192,203,127),
                "B-information":(0,0,203,127),
                "B-name":(255,255,0,127),
                "B-rank":(255,0,0,127),
                "B-birth_date":(255,166,0,127),
                "B-nationality":(148,100,10,127),
                "B-death_date":(128,181,31,127),
                "B-funeral_date":(43,130,14,127),
                "B-grave_location":(21,191,186,127),
                "B-grave_id":(8,20,156,127),
                "B-information_source":(108,118,235,127),
                "B-death_book":(184,108,235,127),
    }

    result_folder = args.input_file.split("/")[-1].split(".")[0]
    

    for i in range(len(results)):
        image_name = results[i]["image_path"].split('/')[-1]

        with Image.open(results[i]["image_path"]) as im:
                draw = ImageDraw.Draw(im, "RGBA")
                
        font = ImageFont.load_default(size=20)
        
        for j in range(len(results[i]['bboxes'])):
            
            if results[i]["ner_tags"][j]!='O':
                draw.rectangle(results[i]['bboxes'][j],fill=label2color_fill[results[i]['ner_tags'][j]])                
                draw.text((results[i]['bboxes'][j][0] + 10, results[i]['bboxes'][j][1]-25), text=results[i]['ner_tags'][j], fill=label2color_fill[results[i]['ner_tags'][j]], font=font)
        
        if args.save_images:
            if args.save_path:
                out_dir = Path(args.save_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / image_name
                im.save(out_path)
            else:
                out_dir = Path(f"out/form_images/{result_folder}")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / image_name
                im.save(out_path)
            print(f"saved to {out_path}")
        else:
            im.show()
            

if __name__ == "__main__":
    args = parse_args()
    main(args)