import numpy as np
import csv
import os
import json
from pathlib import Path
from collections import OrderedDict


def convert_file(file_in, num_type):
    sig = [[num_type(ele) for ele in file] for file in file_in]
    return sig

def get_coresponding_signatures(signatures_list, labels_list, items_list):
    signatures_list_out = list()
    for idx, label in enumerate(labels_list):
        if label in items_list:
            signatures_list_out.append(signatures_list[idx])
    return signatures_list_out

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



