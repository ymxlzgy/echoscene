import json
import yaml


def load_semantic_scene_graphs_custom(yml_relationships, color_palette, rel_label_to_id, with_manipuation=False):
    scene_graphs = {}

    graphs = yaml.load(open(yml_relationships, 'r'))
    for scene_id, scene in graphs['Scenes'].items():

        scene_graphs[str(scene_id)] = {}
        scene_graphs[str(scene_id)]['objects'] = []
        scene_graphs[str(scene_id)]['relationships'] = []
        scene_graphs[str(scene_id)]['node_mask'] = [1] * len(scene['nodes'])
        scene_graphs[str(scene_id)]['edge_mask'] = [1] * len(scene['relships'])

        for (i, n) in enumerate(scene['nodes']):
            obj_item = {'ply_color': color_palette[i%len(color_palette)],
                        'id': str(i),
                        'label': n}
            scene_graphs[str(scene_id)]['objects'].append(obj_item)
        for r in scene['relships']:
            rel_4 = [r[0], r[1], rel_label_to_id[r[2]], r[2]]
            scene_graphs[str(scene_id)]['relationships'].append(rel_4)
        counter = len(scene['nodes'])
        if with_manipuation:
            for m in scene['manipulations']:
                if m[1] == 'add':
                    # visualize an addition
                    # ['chair', 'add', [[2, 'standing on'], [1, 'left']]]
                    obj_item = {'ply_color': color_palette[counter%len(color_palette)],
                                'id': str(counter),
                                'label': m[0]}
                    scene_graphs[str(scene_id)]['objects'].append(obj_item)

                    scene_graphs[str(scene_id)]['node_mask'].append(0)
                    for mani_rel in m[2]:
                        rel_4 = [counter, mani_rel[0], rel_label_to_id[mani_rel[1]], mani_rel[1]]
                        scene_graphs[str(scene_id)]['relationships'].append(rel_4)
                        scene_graphs[str(scene_id)]['edge_mask'].append(0)
                    counter += 1
                if m[1] == 'rel':
                    # visualize changes in the relationship
                    for (rid, r) in enumerate(scene_graphs[str(scene_id)]['relationships']):
                        s, o, p, l = r
                        if isinstance(m[2][3], list):
                            # ['', 'rel', [0, 1, 'right', [0, 1, 'left']]]
                            if s == m[2][0] and o == m[2][1] and l == m[2][2] and s == m[2][3][0] and o == m[2][3][1]:
                                # a change on the SAME (s, o) pair, indicate the change
                                scene_graphs[str(scene_id)]['edge_mask'][rid] = 0
                                scene_graphs[str(scene_id)]['relationships'][rid][3] = m[2][2] + '->' + m[2][3][2]
                                scene_graphs[str(scene_id)]['relationships'][rid][2] = rel_label_to_id[m[2][3][2]]
                                break
                            elif s == m[2][0] and o == m[2][1] and l == m[2][2]:
                                # overwrite this edge with a new pair (s,o)
                                del scene_graphs[str(scene_id)]['edge_mask'][rid]
                                del scene_graphs[str(scene_id)]['relationships'][rid]
                                scene_graphs[str(scene_id)]['edge_mask'].append(0)
                                new_edge = [m[2][3][0], m[2][3][1], rel_label_to_id[m[2][3][2]], m[2][3][2]]
                                scene_graphs[str(scene_id)]['relationships'].append(new_edge)
                        else:
                            # ['', 'rel', [0, 1, 'right', 'left']]
                            if s == m[2][0] and o == m[2][1] and l == m[2][2]:
                                scene_graphs[str(scene_id)]['edge_mask'][rid] = 0
                                scene_graphs[str(scene_id)]['relationships'][rid][3] = m[2][2] + '->' + m[2][3]
                                scene_graphs[str(scene_id)]['relationships'][rid][2] = rel_label_to_id[m[2][3]]
                                break

    return scene_graphs


def load_semantic_scene_graphs(json_relationships, json_objects):
    scene_graphs_obj = {}

    with open(json_objects, "r") as read_file:
        data = json.load(read_file)
        for s in data["scans"]:
            scan = s["scan"]
            objs = s['objects']
            scene_graphs_obj[scan] = {}
            scene_graphs_obj[scan]['scan'] = scan
            scene_graphs_obj[scan]['objects'] = []
            for obj in objs:
                scene_graphs_obj[scan]['objects'].append(obj)
    scene_graphs = {}
    with open(json_relationships, "r") as read_file:
        data = json.load(read_file)
        for s in data["scans"]:
            scan = s["scan"]
            split = str(s["split"])
            if scan + "_" + split not in scene_graphs:
                scene_graphs[scan + "_" + split] = {}
                scene_graphs[scan + "_" + split]['objects'] = []
                print("WARNING: no objects for this scene")
            scene_graphs[scan + "_" + split]['relationships'] = []
            for k in s["objects"].keys():
                ob = s['objects'][k]
                for i,o in enumerate(scene_graphs_obj[scan]['objects']):
                    if o['id'] == k:
                        inst = i
                        break
                scene_graphs[scan + "_" + split]['objects'].append(scene_graphs_obj[scan]['objects'][inst])
            for rel in s["relationships"]:
                scene_graphs[scan + "_" + split]['relationships'].append(rel)
    return scene_graphs


def read_relationships(read_file):
    relationships = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            relationship = line.rstrip().lower() 
            relationships.append(relationship) 
    return relationships
