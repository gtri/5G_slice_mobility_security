import json
import argparse 
from pathlib import Path
import sys

def mergeJson(files, savefile):
    result = list()
    for f1 in files:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    with open(savefile, 'w') as output_file:
        json.dump(result, output_file)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Process raw ves messages and UE data samples into csv datasets.')
    parser.add_argument('json_list', metavar='List of json files', type=list, nargs=1,
                        help='List of json files to combine')
    parser.add_argument('savefile', metavar='File to save out to', type=Path, nargs=1,
                        help='List of json files to combine')

    files = list(sys.argv[1])
    savefile = Path(sys.argv[2])

    mergeJson(files, savefile)

