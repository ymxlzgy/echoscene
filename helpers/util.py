import os

import torch
import numpy as np
import trimesh
import json
import glob
import random
import pytorch3d
from pytorch3d.structures import Meshes

from model.diff_utils.util_3d import sdf_to_mesh


class _CustomDataParallel(torch.nn.Module):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__()
        self.model = torch.nn.DataParallel(model).cuda()
        #self.model = model.cuda()

    def forward(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        #return getattr(self.model, name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

def sample_points(points_list, num):
    resampled_point_clouds = []
    for point_cloud in points_list:
        n_points = point_cloud.size(0)

        if n_points >= num:
            random_indices = torch.randperm(n_points)[:num]
            resampled_point_cloud = point_cloud[random_indices]
        else:
            random_indices = torch.randint(n_points, size=(num,))
            resampled_point_cloud = point_cloud[random_indices]

        resampled_point_clouds.append(resampled_point_cloud)
    return resampled_point_clouds

def get_cross_prod_mat(pVec_Arr):
    """ Convert pVec_Arr of shape (3) to its cross product matrix
    """
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def params_to_8points(box, degrees=False):
    """ Given bounding box as 7 parameters: w, l, h, cx, cy, cz, z, compute the 8 corners of the box
    """
    w, l, h, cx, cy, cz, z = box
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append([w.item()/2 * i, l.item()/2 * j, h.item()/2 * k])
    points = np.asarray(points)
    points = (get_rotation(z.item(), degree=degrees) @ points.T).T
    points += np.expand_dims(np.array([cx.item(), cy.item(), cz.item()]), 0)
    return points

def get_closest_furniture_to_box(box_dict, query_size):

    mses = {}
    lhw = np.array(list(box_dict.values()))
    ids = np.array(list(box_dict.keys()))
    mses = np.sum((lhw - query_size.detach().cpu().numpy()) ** 2, axis=-1)
    id_min = np.argmin(mses)

    # for i, oi in box_dict.items():
    #     l, h, w = oi[0], oi[1], oi[2]
    #     vol = l * h * w
    print("id: ", ids[id_min], np.min(mses))
    return ids[id_min]


def get_textured_objects_v2(boxes, datasize, cat_ids, classes, mesh_dir, render_boxes=False, colors=None, without_lamp=False):
    os.makedirs(mesh_dir,exist_ok=True)
    bbox_file = "/home/ymxlzgy/code/graphto3d_v2/GT/3dfront/cat_jid_trainval.json" if datasize == 'large' else "/home/ymxlzgy/code/graphto3d_v2/GT/3dfront/cat_jid_trainval_small.json"
    colors = iter(colors)
    with open(bbox_file, "r") as read_file:
        box_data = json.load(read_file)
    lamp_mesh_list = []
    trimesh_meshes = []
    raw_meshes = []
    model_base_path = "/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FUTURE-model"
    instance_id = 1
    for j in range(0, boxes.shape[0]):
        query_size = boxes[j, 0:3]
        query_label = classes[cat_ids[j]].strip('\n')
        if query_label == '_scene_' or query_label == 'floor':
            continue
        print('cat: ', query_label)
        furniture_id = get_closest_furniture_to_box(
            box_data[query_label], query_size
        )

        model_path = os.path.join(model_base_path,furniture_id,"raw_model.obj")
        texture_path = os.path.join(model_base_path, furniture_id, "texture.png")
        color = next(colors)

        # Load the furniture and scale it as it is given in the dataset
        tr_mesh = trimesh.load(model_path, force="mesh")
        tr_mesh = trimesh.Trimesh(vertices=tr_mesh.vertices, faces=tr_mesh.faces, process=False)
        tr_mesh.visual.vertex_colors = color
        tr_mesh.visual.face_colors = color
        raw_meshes.append(tr_mesh.copy())

        tr_mesh.export(os.path.join(mesh_dir, query_label+'_'+str(cat_ids[j])+'_'+str(instance_id)+".obj"))
        instance_id += 1
        # tr_mesh.visual.material.image = Image.open(texture_path)
        theta = boxes[j, -1].item() * (np.pi / 180)
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        t = boxes[j, 3:6].detach().cpu().numpy()
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + t
        trimesh_meshes.append(tr_mesh)
        if query_label == 'lamp' and without_lamp:
            lamp_mesh_list.append(trimesh_meshes.pop())
        if render_boxes:
            box_points = params_to_8points_3dfront(boxes[j], degrees=True)
            trimesh_meshes.append(create_bbox_marker(box_points, tube_radius=0.006, color=color))


    return lamp_mesh_list, trimesh_meshes, raw_meshes

def get_bbox(boxes, cat_ids, classes, colors, without_lamp=False):
    trimesh_meshes = []
    colors = iter(colors)
    lamp_mesh_list=[]
    for j in range(0, boxes.shape[0]):
        query_label = classes[cat_ids[j]].strip('\n')
        if query_label == '_scene_' or query_label == 'floor':
            continue
        box_points = params_to_8points_3dfront(boxes[j], degrees=True)
        trimesh_meshes.append(create_bbox_marker(box_points, tube_radius=0.02, color=next(colors)))
        # if query_label == 'nightstand':
        #     trimesh_meshes.pop()
        if query_label == 'lamp' and without_lamp:
            lamp_mesh_list.append(trimesh_meshes.pop())


    return lamp_mesh_list, trimesh_meshes

def fit_shapes_to_box_v2(obj, box, degrees=False):
    l, h, w, px, py, pz, angle = box
    if isinstance(l, torch.Tensor):
        l, h, w, px, py, pz, angle = l.item(), h.item(), w.item(), px.item(), py.item(), pz.item(), angle.item()
    box_points = []
    for i in [-1, 1]:
        for j in [0, 1]:
            for k in [-1, 1]:
                box_points.append([l / 2 * i, h * j, w / 2 * k])

    bounding_box = obj.bounding_box
    bottom_center = bounding_box.bounds[0] + (bounding_box.extents / 2)
    bottom_center[1] = bounding_box.bounds[0][1]
    rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi/2, [0,1,0])
    translation_matrix = trimesh.transformations.translation_matrix(-bottom_center)
    transform = np.dot(translation_matrix, rotation_matrix)
    obj.apply_transform(transform)

    R = get_rotation_3dfront(angle, degree=degrees)
    R_inv = np.linalg.inv(R)
    t = np.array([px, py, pz])
    T = np.concatenate((R_inv,t.reshape(-1,1)),axis=1)
    T = np.concatenate((T,np.array([0,0,0,1]).reshape(1,-1)),axis=0)
    vertices = np.array(obj.vertices)
    shape_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    obj = obj.apply_scale(1 / shape_size)
    obj = obj.apply_scale([l, h, w])
    obj = obj.apply_transform(T)
    box_points = np.asarray(box_points)
    box_points = box_points.dot(R)
    box_points += np.expand_dims(t, 0)
    return box_points, obj

def fit_shapes_to_box_v1(obj, box, degrees=False):
    l, h, w, px, py, pz, angle = box
    box_points = []
    for i in [-1, 1]:
        for j in [0, 1]:
            for k in [-1, 1]:
                box_points.append([l.item() / 2 * i, h.item() * j, w.item() / 2 * k])

    bounding_box = obj.bounding_box
    bottom_center = bounding_box.bounds[0] + (bounding_box.extents / 2)
    bottom_center[1] = bounding_box.bounds[0][1]
    rotation_matrix = trimesh.transformations.rotation_matrix(0, [0,1,0])
    translation_matrix = trimesh.transformations.translation_matrix(-bottom_center)
    transform = np.dot(translation_matrix, rotation_matrix)
    obj.apply_transform(transform)

    R = get_rotation_3dfront(angle.item(), degree=degrees)
    R_inv = np.linalg.inv(R)
    t = np.array([px.item(), py.item(), pz.item()])
    T = np.concatenate((R_inv,t.reshape(-1,1)),axis=1)
    T = np.concatenate((T,np.array([0,0,0,1]).reshape(1,-1)),axis=0)
    vertices = np.array(obj.vertices)
    shape_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    obj = obj.apply_scale(1 / shape_size)
    obj = obj.apply_scale([l.item(), h.item(), w.item()])
    obj = obj.apply_transform(T)
    box_points = np.asarray(box_points)
    box_points = box_points.dot(R)
    box_points += np.expand_dims(t, 0)
    return box_points, obj

def trimeshes_to_pytorch3d(meshes):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    verts_list = []
    faces_list = []
    textures_list = []
    for mesh in meshes:
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)   # [V, 3]
        faces = torch.tensor(mesh.faces, dtype=torch.int64).to(device)   # [F, 3]
        verts_list.append(vertices)
        faces_list.append(faces)
        vertex_colors = torch.tensor(mesh.visual.vertex_colors, dtype=torch.float32).to(device)  # [V, 4]
        vertex_colors = vertex_colors[:, :3] / 255.
        textures_list.append(vertex_colors)
    textures=pytorch3d.renderer.Textures(verts_rgb=textures_list)


    pytorch3d_meshes = Meshes(verts=verts_list, faces=faces_list, textures=textures)

    return pytorch3d_meshes

def normalize_py3d_meshes(meshes):
    centers = meshes.verts_packed().mean(dim=0)
    verts_list_centered = [verts - centers for verts in meshes.verts_list()]

    verts_list_normalized = []
    for verts in verts_list_centered:
        max_distance = verts.norm(p=2, dim=1).max()
        verts_normalized = verts / max_distance
        verts_list_normalized.append(verts_normalized)

    normalized_meshes = Meshes(
        verts=verts_list_normalized,
        faces=meshes.faces_list(),
        textures=meshes.textures,
    )

    return normalized_meshes

def pytorch3d_to_trimesh(pytorch3d_mesh):
    trimesh_verts = pytorch3d_mesh.verts_list()[0].cpu().numpy()
    trimesh_faces = pytorch3d_mesh.faces_list()[0].cpu().numpy()
    trimesh_normals = pytorch3d_mesh.verts_normals_list()[0].cpu().numpy()
    tri_mesh = trimesh.Trimesh(vertices=trimesh_verts, faces=trimesh_faces, process=False)
    tri_mesh.vertex_normals = trimesh_normals
    tri_mesh.invert()
    return tri_mesh

def get_generated_models_v1(boxes, shapes, cat_ids, mesh_dir, classes, render_boxes=False, colors=None, without_lamp=False):
    colors = iter(colors)
    trimesh_meshes = iter(shapes)
    obj_list = []
    lamp_mesh_list = []
    raw_obj_list = []
    instance_id = 1
    for j in range(0, boxes.shape[0]):
        query_label = classes[cat_ids[j]].strip('\n')
        if query_label == '_scene_' or query_label == 'floor':
            continue
        obj = next(trimesh_meshes)
        color = next(colors)
        obj.visual.vertex_colors = color
        obj.visual.face_colors = color
        raw_obj_list.append(obj.copy())
        obj.export(os.path.join(mesh_dir, query_label + '_' + str(cat_ids[j]) + "_" + str(instance_id) + ".obj"))
        instance_id += 1

        box_points, obj = fit_shapes_to_box_v1(obj, boxes[j], degrees=True)
        obj_list.append(obj)
        if query_label == 'lamp' and without_lamp:
            lamp_mesh_list.append(obj_list.pop())


        if render_boxes:
            obj_list.append(create_bbox_marker(box_points, color=color))
    return lamp_mesh_list, obj_list, raw_obj_list

def get_generated_models_v2(boxes, shapes, cat_ids, classes, mesh_dir, render_boxes=False, colors=None, without_lamp=False):
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_gen = sdf_to_mesh(shapes,render_all=True)
    colors = iter(colors)
    trimesh_meshes = iter([pytorch3d_to_trimesh(mesh) for mesh in mesh_gen])
    obj_list = []
    lamp_mesh_list = []
    raw_obj_list = []
    instance_id = 1
    for j in range(0, boxes.shape[0]):
        render_boxes_ = render_boxes
        query_label = classes[cat_ids[j]].strip('\n')
        if query_label == '_scene_' or query_label == 'floor':
            continue
        obj = next(trimesh_meshes)
        color = next(colors)
        obj.visual.vertex_colors = color
        obj.visual.face_colors = color
        raw_obj_list.append(obj.copy())
        obj.export(os.path.join(mesh_dir, query_label + '_' + str(cat_ids[j]) + "_" + str(instance_id)+".obj"))
        instance_id += 1

        box_points, obj = fit_shapes_to_box_v2(obj, boxes[j], degrees=True)
        obj_list.append(obj)
        # if query_label == 'bed':
        #     obj.export('/media/ymxlzgy/Data/asset/bedv2.glb')
        # if query_label == 'nightstand':
        #     obj_list.pop()
        #     render_boxes_ = False
        if query_label == 'lamp' and without_lamp:
            lamp_mesh_list.append(obj_list.pop())

        if render_boxes_:
            obj_list.append(create_bbox_marker(box_points, tube_radius=0.006, color=color))
    return lamp_mesh_list, obj_list, raw_obj_list


def get_sdfusion_models(boxes, cat_ids, classes, mesh_dir, render_boxes=False, colors=None, no_stool=False, without_lamp=False):
    os.makedirs(mesh_dir, exist_ok=True)
    sdfusion_model_path = "/media/ymxlzgy/Data/Dataset/3D-FRONT/txt2shape_results_latest"
    mapping_full2simple = None
    obj_list = []
    colors = iter(colors)
    lamp_mesh_list = []
    raw_obj_list = []
    instance_id = 1
    for j in range(0, boxes.shape[0]):
        query_label = classes[cat_ids[j]].strip('\n')
        if query_label == '_scene_' or query_label == 'floor':
            continue
        color = next(colors)
        folder_path = os.path.join(sdfusion_model_path, query_label) if mapping_full2simple == None \
            else os.path.join(sdfusion_model_path, mapping_full2simple[query_label])
        a=random.randint(0, 100)
        print("seed: ", a)
        random.seed(a)
        model_path = random.choice(glob.glob(os.path.join(folder_path,'*.ply')))
        if no_stool and query_label == 'chair':
            assert mapping_full2simple == None
            model_path2 = random.choice(glob.glob(os.path.join(sdfusion_model_path,'stool','*.ply')))
            model_path = random.choice([model_path, model_path2])

        obj = trimesh.load(model_path)
        obj.invert()
        obj.visual.vertex_colors = color
        obj.visual.face_colors = color
        raw_obj_list.append(obj.copy())
        obj.export(os.path.join(mesh_dir, query_label + '_' + str(cat_ids[j]) + "_" + str(instance_id)+".obj"))
        instance_id += 1

        box_points, obj = fit_shapes_to_box_v2(obj, boxes[j], degrees=True)
        obj_list.append(obj)
        if query_label == 'lamp' and without_lamp:
            lamp_mesh_list.append(obj_list.pop())

        if render_boxes:
            obj_list.append(create_bbox_marker(box_points, color=color))

    return lamp_mesh_list, obj_list, raw_obj_list


def params_to_8points_3dfront(box, degrees=False):
    """ Given bounding box as 7 parameters: l, h, w, cx, cy, cz, z, compute the 8 corners of the box
    """
    l, h, w, px, py, pz, angle = box
    points = []
    for i in [-1, 1]:
        for j in [0, 1]:
            for k in [-1, 1]:
                points.append([l.item()/2 * i, h.item() * j, w.item()/2 * k])
    points = np.asarray(points)
    points = points.dot(get_rotation_3dfront(angle.item(), degree=degrees))
    points += np.expand_dims(np.array([px.item(), py.item(), pz.item()]), 0)
    return points

def create_bbox_marker(corner_points, color=[0, 0, 255], tube_radius=0.002, sections=4):
    """Create a 3D mesh visualizing a bbox. It consists of 12 cylinders.

    Args:
        corner_points
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 4.

    Returns:
        trimesh.Trimesh: A mesh.
    """
    edges = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    bbox_edge_list = []
    for edge in edges:
        bbox_edge = trimesh.creation.cylinder(radius=tube_radius,sections=sections,segment=[corner_points[edge[0]],corner_points[edge[1]]])
        bbox_edge_list.append(bbox_edge)

    tmp = trimesh.util.concatenate(bbox_edge_list)
    tmp.visual.face_colors = color

    # z axis to x axis
    # R = np.array([[0,0,1],[1,0,0],[0,1,0]]).reshape(3,3)
    # t =  np.array([0, 0, -1.12169998e-01]).reshape(3,1)
    #
    # T = np.r_[np.c_[np.eye(3), t], [[0, 0, 0, 1]]]
    # tmp.apply_transform(T)

    return tmp


def params_to_8points_no_rot(box):
    """ Given bounding box as 6 parameters (without rotation): w, l, h, cx, cy, cz, compute the 8 corners of the box.
        Works when the box is axis aligned
    """
    w, l, h, cx, cy, cz = box
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append([w.item()/2 * i, l.item()/2 * j, h.item()/2 * k])
    points = np.asarray(points)
    points += np.expand_dims(np.array([cx.item(), cy.item(), cz.item()]), 0)
    return points


def fit_shapes_to_box(box, shape, withangle=True):
    """ Given normalized shape, transform it to fit the input bounding box.
        Expects denormalized bounding box with optional angle channel in degrees
        :param box: tensor
        :param shape: tensor
        :param withangle: boolean
        :return: transformed shape
    """
    box = box.detach().cpu().numpy()
    shape = shape.detach().cpu().numpy()
    if withangle:
        w, l, h, cx, cy, cz, z = box
    else:
        w, l, h, cx, cy, cz = box
    # scale
    shape_size = np.max(shape, axis=0) - np.min(shape, axis=0)
    shape = shape / shape_size
    shape *= box[:3]
    if withangle:
        # rotate
        shape = (get_rotation(z, degree=True).astype("float32") @ shape.T).T
    # translate
    shape += [cx, cy, cz]

    return shape

# TODO
def refineBoxes(boxes, objs, triples, relationships, vocab):
    for idx in range(len(boxes)):
      child_box = boxes[idx]
      w, l, h, cx, cy, cz = child_box
      for t in triples:
         if idx == t[0] and relationships[t[1]] in ["supported by", "lying on", "standing on"]:
            parent_idx = t[2]
            cat = vocab['object_idx_to_name'][objs[parent_idx]].replace('\n', '')
            if cat != 'floor':
                continue
            parent_box = boxes[parent_idx]
            base = parent_box[5] + 0.0125

            new_bottom = base
            # new_h = cz + h / 2 - new_bottom
            new_cz = new_bottom + h / 2
            shift = new_cz - cz
            boxes[idx][:] = [w, l, h, cx, cy, new_cz]

            # fix adjusmets
            for t_ in triples:
                if t_[2] == t[0] and relationships[t_[1]] in ["supported by", "lying on", "standing on"]:
                    cat = vocab['object_idx_to_name'][t_[2]].replace('\n', '')
                    if cat != 'floor':
                        continue

                    w_, l_, h_, cx_, cy_, cz_ = boxes[t_[0]]
                    boxes[t_[0]][:] = [w_, l_, h_, cx_, cy_, cz_ + shift]
    return boxes


def get_rotation(z, degree=True):
    """ Get rotation matrix given rotation angle along the z axis.
    :param z: angle of z axos rotation
    :param degree: boolean, if true angle is given in degrees, else in radians
    :return: rotation matrix as np array of shape[3,3]
    """
    if degree:
        z = np.deg2rad(z)
    rot = np.array([[np.cos(z), -np.sin(z),  0],
                    [np.sin(z),  np.cos(z),  0],
                    [        0,          0,  1]])
    return rot

def get_rotation_3dfront(y, degree=True):
    if degree:
        y = np.deg2rad(y)
    rot = np.array([[np.cos(y),     0,  -np.sin(y)],
                    [       0 ,     1,           0],
                    [np.sin(y),     0,   np.cos(y)]])
    return rot


def normalize_box_params(box_params, file=None, scale=3):
    """ Normalize the box parameters for more stable learning utilizing the accumulated dataset statistics

    :param box_params: float array of shape [7] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :return: normalized box parameters array of shape [7]
    """
    assert file is not None
    if file == None:
        mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955])
        std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435])
    else:
        stats = np.loadtxt(file)
        mean, std = stats[0], stats[1]

    return scale * ((box_params - mean) / std)


def denormalize_box_params(box_params, file=None, scale=3, params=7):
    """ Denormalize the box parameters utilizing the accumulated dataset statistics

    :param box_params: float array of shape [params] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :param params: number of bounding box parameters. Expects values of either 6 or 7. 6 omits the angle
    :return: denormalized box parameters array of shape [params]
    """
    assert file is not None
    if file == None:
        if params == 6:
            mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847])
            std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753])
        elif params == 7:
            mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955])
            std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435])
        else:
            raise NotImplementedError
    else:
        stats = np.loadtxt(file)
        if params == 6:
            mean, std = stats[0][:6], stats[1][:6]
        else:
            mean, std = stats[0], stats[1]
    return (box_params * std) / scale + mean


def batch_torch_denormalize_box_params(box_params, file=None, scale=3, params=7):
    """ Denormalize the box parameters utilizing the accumulated dataset statistics

    :param box_params: float tensor of shape [N, 6] containing the 6 box parameters, where N is the number of boxes
    :param scale: float scalar that scales the parameter distribution
    :return: float tensor of shape [N, 6], the denormalized box parameters
    """
    assert file is not None
    if file == None:
        if params == 6:
            mean = torch.from_numpy(np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847]).reshape(1,-1)).cuda()
            std = torch.from_numpy(np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753]).reshape(1,-1)).cuda()
        elif params == 7:
            mean = torch.from_numpy(np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955]).reshape(1,-1)).cuda()
            std = torch.from_numpy(np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435]).reshape(1,-1)).cuda()
        else:
            raise NotImplementedError
    else:
        stats = np.loadtxt(file)
        if params == 6:
            mean, std = torch.from_numpy(stats[0][:6].reshape(1,-1)).cuda(), torch.from_numpy(stats[1][:6].reshape(1,-1)).cuda()
        else:
            mean, std = torch.from_numpy(stats[0].reshape(1,-1)).cuda(), torch.from_numpy(stats[1].reshape(1,-1)).cuda()

    return (box_params * std) / scale + mean


def bool_flag(s):
    """Helper function to make argparse work with the input True and False.
    Otherwise it reads it as a string and is always set to True.

    :param s: string input
    :return: the boolean equivalent of the string s
    """
    if s == '1' or s == 'True':
      return True
    elif s == '0' or s == 'False':
      return False
    msg = 'Invalid value "%s" for bool flag (should be 0, False or 1, True)'
    raise ValueError(msg % s)
