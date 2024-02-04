#!/usr/bin/python

import numpy as np
import torch
from scipy.spatial import ConvexHull
from helpers.util import denormalize_box_params
from helpers.util import fit_shapes_to_box
from cmath import rect, phase

def close_dis(corners1,corners2):
    dist = -2 * np.matmul(corners1, corners2.transpose())
    dist += np.sum(corners1 ** 2, axis=-1)[:, None]
    dist += np.sum(corners2 ** 2, axis=-1)[None, :]
    dist = np.sqrt(dist)
    return np.min(dist)

def cal_l2_distance(point_1, point_2):
    return np.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)

def angular_distance(a, b):
    a %= 360.
    b %= 360.

    va = np.matmul(rot2d(a), [1, 0])
    vb = np.matmul(rot2d(b), [1, 0])
    return anglebetween2vecs(va, vb) % 360.


def anglebetween2vecs(va, vb):
    rad = np.arccos(np.clip(np.dot(va, vb), -1, 1))
    return np.rad2deg(rad)


def rot2d(degrees):
    rad = np.deg2rad(degrees)
    return np.asarray([[np.cos(rad), -np.sin(rad)],
                       [np.sin(rad), np.cos(rad)]])


def estimate_angular_mean(deg):
    return np.rad2deg(phase(np.sum(rect(1, np.deg2rad(d)) for d in deg)/len(deg))) % 360.


def estimate_angular_std(degs):
    m = estimate_angular_mean(degs)
    std = np.sqrt(np.sum([angular_distance(d, m)**2 for d in degs]) / len(degs))
    return std


def denormalize(box_params, file=None, with_norm=True):
    if with_norm:
        return denormalize_box_params(box_params, file=file, params=box_params.shape[0])
    else:
        return box_params


def validate_constrains(triples, pred_boxes, gt_boxes, keep, vocab, accuracy, file_dist=None, with_norm=True,
                        strict=True, overlap_threshold=0.3):

    param6 = pred_boxes.shape[1] == 6
    layout_boxes = pred_boxes

    for [s, p, o] in triples:
        if keep is None:
            box_s = denormalize(layout_boxes[s.item()].cpu().detach().numpy(), file=file_dist,  with_norm=with_norm)
            box_o = denormalize(layout_boxes[o.item()].cpu().detach().numpy(), file=file_dist, with_norm=with_norm)
        else:
            if keep[s.item()] == 1 and keep[o.item()] == 1: # if both are unchanged we evaluate the normal constraints
                box_s = denormalize(layout_boxes[s.item()].cpu().detach().numpy(), file=file_dist, with_norm=with_norm)
                box_o = denormalize(layout_boxes[o.item()].cpu().detach().numpy(), file=file_dist, with_norm=with_norm)
            else:
                continue

        if vocab["pred_idx_to_name"][p.item()][:-1] == "left":
            # z
            if box_s[5] - box_o[5] > -0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['left'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['left'].append(1)
                accuracy['total'].append(1)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "right":
            if box_s[5] - box_o[5] < 0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['right'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['right'].append(1)
                accuracy['total'].append(1)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "front":
            if box_s[3] - box_o[3] < -0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['front'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['front'].append(1)
                accuracy['total'].append(1)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "behind":
            if box_s[3] - box_o[3] > 0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['behind'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['behind'].append(1)
                accuracy['total'].append(1)
        # bigger than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "bigger than":
            sub_volume = box_s[0] * box_s[1] * box_s[2]
            obj_volume = box_o[0] * box_o[1] * box_o[2]
            if (sub_volume - obj_volume) / sub_volume < 0.15:
                accuracy['bigger'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['bigger'].append(1)
                accuracy['total'].append(1)
        # smaller than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "smaller than":
            sub_volume = box_s[0] * box_s[1] * box_s[2]
            obj_volume = box_o[0] * box_o[1] * box_o[2]
            if (sub_volume - obj_volume) / sub_volume > -0.15:
                accuracy['smaller'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['smaller'].append(1)
                accuracy['total'].append(1)
        # higher than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "taller than":
            absheight_s = box_s[4]+box_s[1]
            absheight_o = box_o[4]+box_o[1]
            if (absheight_s - absheight_o) / absheight_s < 0.1:
                accuracy['taller'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['taller'].append(1)
                accuracy['total'].append(1)
        # lower than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "shorter than":
            absheight_s = box_s[4] + box_s[1]
            absheight_o = box_o[4] + box_o[1]
            if (absheight_s - absheight_o) / absheight_s > -0.1:
                accuracy['shorter'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['shorter'].append(1)
                accuracy['total'].append(1)

        # standing on
        if vocab["pred_idx_to_name"][p.item()][:-1] == "standing on":
            if np.abs(box_s[4] - box_o[4]) < 0.04:
                accuracy['standing on'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['standing on'].append(0)
                accuracy['total'].append(0)
        # close by
        if vocab["pred_idx_to_name"][p.item()][:-1] == "close by":
            corners_s = corners_from_box(box_s, param6, with_translation=True)
            corners_o = corners_from_box(box_o, param6, with_translation=True)
            c_dist1 = close_dis(corners_s, corners_o)
            if c_dist1 > 0.45:
                accuracy['close by'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['close by'].append(1)
                accuracy['total'].append(1)

        # symmetrical to
        if vocab["pred_idx_to_name"][p.item()][:-1] == "symmetrical to":
            sub_center_in_scene_flip_x = [-box_s[3], box_s[5]]
            sub_center_in_scene_flip_z = [box_s[3], -box_s[5]]
            sub_center_in_scene_flip_xz = [-box_s[3], -box_s[5]]
            obj_center_in_scene = [box_o[3], box_o[5]]
            if cal_l2_distance(sub_center_in_scene_flip_xz, obj_center_in_scene) < 0.45 or \
                cal_l2_distance(sub_center_in_scene_flip_x, obj_center_in_scene) < 0.45 or \
                cal_l2_distance(sub_center_in_scene_flip_z, obj_center_in_scene) < 0.45:
                accuracy['symmetrical to'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['symmetrical to'].append(0)
                accuracy['total'].append(0)

    return accuracy


def validate_constrains_changes(triples, pred_boxes, gt_boxes, keep, vocab, accuracy, file_dist=None, with_norm=True,
                                strict=True, overlap_threshold=0.3):

    layout_boxes = pred_boxes

    for [s, p, o] in triples:
        if keep is None:
            box_s = denormalize(layout_boxes[s.item()].cpu().detach().numpy(), file=file_dist, with_norm=with_norm)
            box_o = denormalize(layout_boxes[o.item()].cpu().detach().numpy(), file=file_dist, with_norm=with_norm)
        else:
            if keep[s.item()] == 0 or keep[o.item()] == 0: # if any node is change we evaluate the changes
                box_s = denormalize(layout_boxes[s.item()].cpu().detach().numpy(), file=file_dist, with_norm=with_norm)
                box_o = denormalize(layout_boxes[o.item()].cpu().detach().numpy(), file=file_dist, with_norm=with_norm)
            else:
                continue

        if vocab["pred_idx_to_name"][p.item()][:-1] == "left":
            # z
            if box_s[5] - box_o[5] > -0.05 or (
                    strict and box3d_iou(box_s, box_o, with_translation=True)[
                0] > overlap_threshold):
                accuracy['left'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['left'].append(1)
                accuracy['total'].append(1)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "right":
            if box_s[5] - box_o[5] < 0.05 or (
                    strict and box3d_iou(box_s, box_o, with_translation=True)[
                0] > overlap_threshold):
                accuracy['right'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['right'].append(1)
                accuracy['total'].append(1)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "front":
            if box_s[3] - box_o[3] < -0.05 or (
                    strict and box3d_iou(box_s, box_o, with_translation=True)[
                0] > overlap_threshold):
                accuracy['front'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['front'].append(1)
                accuracy['total'].append(1)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "behind":
            if box_s[3] - box_o[3] > 0.05 or (
                    strict and box3d_iou(box_s, box_o, with_translation=True)[
                0] > overlap_threshold):
                accuracy['behind'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['behind'].append(1)
                accuracy['total'].append(1)
        # bigger than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "bigger than":
            sub_volume = box_s[0] * box_s[1] * box_s[2]
            obj_volume = box_o[0] * box_o[1] * box_o[2]
            if (sub_volume - obj_volume) / sub_volume < 0.15:
                accuracy['bigger'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['bigger'].append(1)
                accuracy['total'].append(1)
        # smaller than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "smaller than":
            sub_volume = box_s[0] * box_s[1] * box_s[2]
            obj_volume = box_o[0] * box_o[1] * box_o[2]
            if (sub_volume - obj_volume) / sub_volume > -0.15:
                accuracy['smaller'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['smaller'].append(1)
                accuracy['total'].append(1)
        # higher than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "taller than":
            absheight_s = box_s[4] + box_s[1]
            absheight_o = box_o[4] + box_o[1]
            if (absheight_s - absheight_o) / absheight_s < 0.1:
                accuracy['taller'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['taller'].append(1)
                accuracy['total'].append(1)
        # lower than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "shorter than":
            absheight_s = box_s[4] + box_s[1]
            absheight_o = box_o[4] + box_o[1]
            if (absheight_s - absheight_o) / absheight_s > -0.1:
                accuracy['shorter'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['shorter'].append(1)
                accuracy['total'].append(1)

        # standing on
        if vocab["pred_idx_to_name"][p.item()][:-1] == "standing on":
            if np.abs(box_s[4] - box_o[4]) < 0.04:
                accuracy['standing on'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['standing on'].append(0)
                accuracy['total'].append(0)
        # close by
        if vocab["pred_idx_to_name"][p.item()][:-1] == "close by":
            corners_s = corners_from_box(box_s, with_translation=True)
            corners_o = corners_from_box(box_o, with_translation=True)
            c_dist1 = close_dis(corners_s, corners_o)
            if c_dist1 > 0.45:
                accuracy['close by'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['close by'].append(1)
                accuracy['total'].append(1)

        # symmetrical to
        if vocab["pred_idx_to_name"][p.item()][:-1] == "symmetrical to":
            sub_center_in_scene_flip_x = [-box_s[3], box_s[5]]
            sub_center_in_scene_flip_z = [box_s[3], -box_s[5]]
            sub_center_in_scene_flip_xz = [-box_s[3], -box_s[5]]
            obj_center_in_scene = [box_o[3], box_o[5]]
            if cal_l2_distance(sub_center_in_scene_flip_xz, obj_center_in_scene) < 0.45 or \
                    cal_l2_distance(sub_center_in_scene_flip_x, obj_center_in_scene) < 0.45 or \
                    cal_l2_distance(sub_center_in_scene_flip_z, obj_center_in_scene) < 0.45:
                accuracy['symmetrical to'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['symmetrical to'].append(0)
                accuracy['total'].append(0)

    return accuracy


def corners_from_box(box, param6=True, with_translation=False):
    # box given as: [l, h, w, px, py, pz, z]
    # l meansures z axis; h measures y axis; w measures x axis.
    # (px, py, pz) is the bottom center
    if param6:
        l, h, w, px, py, pz = box
    else:
        l, h, w, px, py, pz, _ = box

    (tx, ty, tz) = (px, py, pz) if with_translation else (0,0,0)

    x_corners = [w/2,w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2]
    y_corners = [h,h,h,h,0,0,0,0]
    z_corners = [l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,l/2]
    corners_3d = np.dot(np.eye(3), np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + tx
    corners_3d[1,:] = corners_3d[1,:] + ty
    corners_3d[2,:] = corners_3d[2,:] + tz
    corners_3d = np.transpose(corners_3d)

    return corners_3d


def box3d_iou(box1, box2, param6=True, with_translation=False):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is positive Y_h
        corners2: numpy array (8,3), assume up direction is positive Y_h
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    # corner points are in counter clockwise order
    corners1 = corners_from_box(box1, param6, with_translation)
    corners2 = corners_from_box(box2, param6, with_translation)

    rect1 = [(corners1[i,2], corners1[i,0]) for i in range(0,4)]
    rect2 = [(corners2[i,2], corners2[i,0]) for i in range(0,4)]

    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)

    volmin = min(vol1, vol2)

    iou = inter_vol / volmin #(vol1 + vol2 - inter_vol)

    return iou, iou_2d


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

    def computeIntersection():
        dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
        dp = [ s[0] - e[0], s[1] - e[1] ]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return(outputList)


def pointcloud_overlap(pclouds, objs, boxes, angles, triples, vocab, overlap_metric):

    obj_classes = vocab['object_idx_to_name']
    pred_classes = vocab['pred_idx_to_name']
    pair = [(t[0].item(),t[2].item()) for t in triples]
    pred = [t[1].item() for t in triples]
    pair2pred = dict(zip(pair, pred))
    structural = ['floor', 'wall', 'ceiling', '_scene_']
    touching = ['none', 'inside', 'attached to', 'part of', 'cover', 'belonging to', 'build in', 'connected to']
    boxes = torch.cat([boxes.float(), angles.view(-1,1).float()], 1)

    for i in range(len(pclouds) - 1):
        for j in range(i+1, len(pclouds)):
            if obj_classes[objs[i]].split('\n')[0] in structural or \
                    obj_classes[objs[j]].split('\n')[0] in structural:
                # do not consider structural objects
                continue
            if (i, j) in pair2pred.keys() and pred_classes[pair2pred[(i,j)]].split('\n')[0] in touching:
                # naturally expected overlap
                continue
            if (j, i) in pair2pred.keys() and pred_classes[pair2pred[(j,i)]].split('\n')[0] in touching:
                # naturally expected overlap
                continue
            pc1 = fit_shapes_to_box(boxes[i].clone(), pclouds[i].clone())
            pc2 = fit_shapes_to_box(boxes[j].clone(), pclouds[j].clone())
            result = pointcloud_overlap_pair(pc1, pc2)
            overlap_metric.append(result)
    return overlap_metric


def pointcloud_overlap_pair(pc1, pc2):
    from sklearn.neighbors import NearestNeighbors
    all_pc = np.concatenate([pc1, pc2], 0)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
    nbrs.fit(all_pc)
    distances, indices = nbrs.kneighbors(pc1)
    # first neighbour will likely be itself other neighbour is a point from the same pc or the other pc
    # two point clouds are overlaping, when the nearest neighbours of one set are from the other set
    overlap = np.sum(indices >= len(pc1))
    return overlap
