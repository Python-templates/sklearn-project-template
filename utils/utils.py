import numpy as np
import csv
import os
import json
from pathlib import Path
from collections import OrderedDict


def read_csv(data_path):
    with open(data_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)

        data_list = []
        line_count = 0
        for row in csv_reader:
            line_count += 1
            data_list.append(row)
        print(f'Processed {line_count} lines.')
    return data_list

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)



