"""helper file to draw provided bboxes into a picture"""

import utils
import argparse


def parse_args():
    parser = argparse.ArgumentParser(prog="draw_rects", description="draw rects into a image")
    parser.add_argument("--input", default=None, help="input files idk")

def main(args):
    print("HELP")

if __name__ == "__main__":
    args=parse_args()
    main(args)