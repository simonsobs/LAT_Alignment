import argparse
import os
import yaml
from .io import load_tracker

def tracker_txt_to_yaml():
    """
    Convert a text file from the tracker to a tracker yaml.
    This currently puts all points into a single element.

    TODO: Add some smarts to allow for multiple elements.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to tracker txt file to convert")
    parser.add_argument("element", type=str, help="Which element to put this data into")
    args = parser.parse_args()

    data = load_tracker(args.file)
    pt_list = [pt.tolist() for pt in data.points]
    out_dict = {args.element : pt_list}

    out_path = f"{os.path.splitext(args.file)[0]}.yaml"
    with open(out_path, 'w+') as f:
        yaml.dump(out_dict, f)
