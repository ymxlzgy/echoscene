import os

import open3d as o3d
import numpy as np
import trimesh

from helpers.util import fit_shapes_to_box, params_to_8points, params_to_8points_no_rot, params_to_8points_3dfront, get_textured_objects_v2, get_sdfusion_models, get_bbox, get_generated_models_v1, get_generated_models_v2
import json
import torch
import cv2
import pyrender
from pyrender.constants import RenderFlags
from render.lineMesh import LineMesh
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint

from collections import Counter, defaultdict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def analyse():

    dataset_folder = '/mnt/hdd1/3D-FRONT/'
    train_rels = dataset_folder + "relationships_all_trainval.json"
    test_rels = dataset_folder + "relationships_all_test.json"

    with open(train_rels, "r") as read_file:
        trainrel = json.load(read_file)['scans']

    with open(test_rels, "r") as read_file:
        testrel = json.load(read_file)['scans']

    train_obj_counter, train_rel_counter, test_obj_counter, test_rel_counter = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)

    for scan in trainrel:
        rels = scan['relationships']
        for relationship in rels:
            rel = relationship[2] - 1
            train_rel_counter[rel] += 1

    print(sum(train_rel_counter.values()))

    for scan in testrel:
        rels = scan['relationships']
        for relationship in rels:
            rel = relationship[2] - 1
            test_rel_counter[rel] += 1

    print(sum(test_rel_counter.values()))

def analyse_all():

    dataset_folder = '/mnt/hdd1/3D-FRONT/'
    rels = dataset_folder + "relationships_bedroom_trainval.json"
    rels2 = dataset_folder + "relationships_bedroom_test.json"
    # rels = dataset_folder + "relationships_diningroom_all.json"
    #rels = dataset_folder + "relationships_livingroom_all.json"
    obj_mapper = dataset_folder + 'mapping_no_stool.json'
    relationship_labels = ["left", "right", "front", "behind", "close by", "above", "standing on", "bigger than",
                           "smaller than", "taller than", "shorter than", "symmetrical to", "same style as",
                           "same supercat as", "same material as"]

    object_labels = ['_scene_', 'bed', 'bookshelf', 'cabinet', 'chair', 'desk', 'floor', 'lamp', 'nightstand', 'shelf', 'sofa', 'table', 'tv_stand', 'wardrobe']

    color_values = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'yellow', 'teal',
              'gold', 'lime', 'violet']

    with open(rels, "r") as read_file:
        rels_all = json.load(read_file)['scans']

    with open(obj_mapper, "r") as read_file:
        obj_maps = json.load(read_file)

    rel_label_colors = {}
    for i, rel_label in enumerate(relationship_labels):
        rel_label_colors[rel_label] = color_values[i]

    color_values = ['lime', 'brown', 'green', 'yellow', 'teal', 'pink', 'magenta', 'gray', 'purple', 'violet', 'blue', 'red', 'gold', 'orange', 'cyan']
    obj_label_colors = {}
    for i, obj_label in enumerate(object_labels):
        obj_label_colors[obj_label] = color_values[i]

    train_obj_counter, train_rel_counter,= defaultdict(int), defaultdict(int)

    for scan in rels_all:
        objs = scan['objects']
        rels = scan['relationships']
        for relationship in rels:
            rel = relationship[2] - 1
            train_rel_counter[relationship_labels[rel]] += 1

        for _, obj in objs.items():
            mapped_name = obj_maps[obj]
            train_obj_counter[mapped_name] += 1

    with open(rels2, "r") as read_file:
        rels_all2 = json.load(read_file)['scans']
    for scan in rels_all2:
        objs = scan['objects']
        rels = scan['relationships']
        for relationship in rels:
            rel = relationship[2] - 1
            train_rel_counter[relationship_labels[rel]] += 1

        for _, obj in objs.items():
            mapped_name = obj_maps[obj]
            train_obj_counter[mapped_name] += 1

    # Sorting the data in increasing order by values
    sorted_data = sorted(train_rel_counter.items(), key=lambda x: x[1], reverse=True)
    # Extracting the labels and values from the sorted data
    labels = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]

    title = 'Relations'
    fig_title = 'Bedroom_rels.pdf' # 'Diningroom_rels.pdf' 'Livingroom_rels.pdf' 'Bedroom_rels.pdf'
    prepare_rel_chart(labels, values, title, fig_title, label_colors=rel_label_colors)

    # Sorting the data in increasing order by values
    sorted_data = sorted(train_obj_counter.items(), key=lambda x: x[1], reverse=True)
    # Extracting the labels and values from the sorted data
    labels = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]

    title = 'Objects'
    fig_title = 'Bedroom_objs.pdf' #'Bedroom_objs.pdf' 'Diningroom_objs.pdf' 'Livingroom_objs.pdf'
    prepare_rel_chart(labels, values, title, fig_title, label_colors=obj_label_colors, set_log=True)

def analyse_objs():

    cat_file = "/mnt/hdd1/3D-FRONT/cat_jid_trainval_small.json"

    with open(cat_file, "r") as read_file:
        cats = json.load(read_file)

    print("what?")

    obj_counter = defaultdict(int)

    for cat, list in cats.items():
        obj_counter[cat] = len(list)

    dataset_folder = '/mnt/hdd1/3D-FRONT/'
    rels = dataset_folder + "relationships_bedroom_trainval.json"
    rels2 = dataset_folder + "relationships_bedroom_test.json"
    # rels = dataset_folder + "relationships_diningroom_all.json"
    # rels = dataset_folder + "relationships_livingroom_all.json"

    with open(rels, "r") as read_file:
        rels_all = json.load(read_file)['scans']

    with open(rels2, "r") as read_file:
        rels_all2 = json.load(read_file)['scans']

def prepare_rel_chart(labels, values, title, fig_title, label_colors, set_log=False):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # Data for the bar chart
    #labels = ['Label1', 'Label2', 'Label3', 'Label4', 'Label5', 'Label6', 'Label7', 'Label8', 'Label9', 'Label10',
     #         'Label11', 'Label12', 'Label13', 'Label14', 'Label15']
    #values = [10, 15, 7, 12, 9, 13, 8, 5, 11, 6, 14, 3, 9, 4, 8]


    # Creating the horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    # Adjusting the margins
    #plt.subplots_adjust(left=0.15)  # Increase or decrease the left margin as needed

    # Plotting the bar chart with colors from the label_colors dictionary
    for label, value in zip(labels, values):
        ax.barh(label, value, color=label_colors[label])

    # Setting the y-axis scale to logarithmic
    if set_log:
        ax.set_xscale('log')

    # Adding labels and title
    ax.set_xlabel('Occurrence', fontsize=16)
    #ax.set_ylabel('Relations', fontsize=14)
    ax.set_title(title, fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    from matplotlib.ticker import FuncFormatter
    # Displaying the chart
    #plt.show()
    if not set_log:
        # Formatting x-axis tick labels divided by 1000
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x / 1000)))
        ax.set_xlabel(r'Occurrence ($x10^3)$', fontsize=16)

    fig.tight_layout()
    plt.savefig(fig_title, format="pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    #analyse()
    #analyse_all()
    analyse_objs()