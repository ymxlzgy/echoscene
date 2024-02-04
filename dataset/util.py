import numpy as np
from plyfile import PlyData, PlyElement


def read_all_ply(filename):
    """ Reads a PLY file from disk.
    Args:
    filename: string
    
    Returns: np.array, np.array, np.array
    """
    file = open(filename, 'rb')
    plydata = PlyData.read(file)
    points = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).transpose()
    colors = np.stack((plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue'])).transpose()
    try:
        labels = plydata['vertex']['objectId']
    except:
        try:
            labels = plydata['vertex']['label']
        except:
            labels = np.array([])        
    try:
        faces = np.array(plydata['face'].data['vertex_indices'].tolist())
    except:
        faces = np.array([])

    file.close()

    return points, labels, colors, faces


def read_ply(filename, points_only=False):
    """ Reads a PLY file from disk.
    Args:
    filename: string
    
    Returns: np.array, np.array, np.array
    """
    file = open(filename, 'rb')
    plydata = PlyData.read(file)
    points = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).transpose()

    if points_only:
        return points
    try:
        labels = plydata['vertex']['objectId']
    except:
        try:
            labels = plydata['vertex']['label']
        except:
            labels = np.array([])        
    try:
        faces = np.array(plydata['face'].data['vertex_indices'].tolist())
    except:
        faces = np.array([])

    try:
        masks = plydata['vertex']['mask']
    except:
        masks = np.array([])

    file.close()

    return points, labels, faces, masks


def write_ply(filename, points, mask=None, faces=None):
    """ Writes a set of points, optionally with faces, labels and a colormap as a PLY file to disk.
    Args:
    filename: string
    points: np.array
    faces: np.array
    labels: np.array
    colormap: np.array
    """
    colors = [[0, 0, 0], [0, 255, 0], [0, 128, 0], [0, 0, 255]]
    with open(filename, 'w') as file:

        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n' % points.shape[0])
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if mask is not None:
            file.write('property ushort mask\n')
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
        
        if faces is not None:
            file.write('element face %d\n' % faces.shape[0])
            file.write('property list uchar int vertex_indices\n')

        file.write('end_header\n')

        if mask is None:
            for point_i in range(points.shape[0]):
                file.write('%f %f %f\n' % (points[point_i, 0], points[point_i, 1], points[point_i, 2]))
        else:
            for point_i in range(points.shape[0]):
                file.write('%f %f %f %i %i %i % i\n' % (points[point_i, 0], points[point_i, 1], points[point_i, 2], mask[point_i], colors[mask[point_i]][0], colors[mask[point_i]][1], colors[mask[point_i]][2]))

        if faces is not None:
            for face_i in range(faces.shape[0]):
                file.write('3 %d %d %d\n' % (
                    faces[face_i, 0], faces[face_i, 1], faces[face_i, 2]))
