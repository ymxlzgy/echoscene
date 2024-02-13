# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from mmdet3d.utils import array_converter
from mmdet3d.structures.points import BasePoints
try:
    from .base_box3d import BaseInstance3DBoxes
except:
    from base_box3d import BaseInstance3DBoxes

@array_converter(apply_to=('points', 'angles'))
def rotation_3d_in_axis(
    points: Union[np.ndarray, Tensor],
    angles: Union[np.ndarray, Tensor, float],
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor], np.ndarray,
           Tensor]:
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray or Tensor): Points with shape (N, M, 3).
        angles (np.ndarray or Tensor or float): Vector of angles with shape
            (N, ).
        axis (int): The axis to be rotated. Defaults to 0.
        return_mat (bool): Whether or not to return the rotation matrix
            (transposed). Defaults to False.
        clockwise (bool): Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: When the axis is not in range [-3, -2, -1, 0, 1, 2], it
            will raise ValueError.

    Returns:
        Tuple[np.ndarray, np.ndarray] or Tuple[Tensor, Tensor] or np.ndarray or
        Tensor: Rotated points with shape (N, M, 3) and rotation matrix with
        shape (N, 3, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 and \
        points.shape[0] == angles.shape[0], 'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, rot_sin, zeros]),
                torch.stack([-rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, rot_sin]),
                torch.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(
                f'axis should in range [-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, rot_sin]),
            torch.stack([-rot_sin, rot_cos])
        ])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new

# def yaw2local(yaw: Tensor, loc: Tensor) -> Tensor:
#     """Transform global yaw to local yaw (alpha in kitti) in camera
#     coordinates, ranges from -pi to pi.
#
#     Args:
#         yaw (Tensor): A vector with local yaw of each box in shape (N, ).
#         loc (Tensor): Gravity center of each box in shape (N, 3).
#
#     Returns:
#         Tensor: Local yaw (alpha in kitti).
#     """
#     local_yaw = yaw - torch.atan2(loc[:, 0], loc[:, 2])
#     larger_idx = (local_yaw > np.pi).nonzero(as_tuple=False)
#     small_idx = (local_yaw < -np.pi).nonzero(as_tuple=False)
#     if len(larger_idx) != 0:
#         local_yaw[larger_idx] -= 2 * np.pi
#     if len(small_idx) != 0:
#         local_yaw[small_idx] += 2 * np.pi
#
#     return local_yaw

class Threedfront3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances.

    Coordinates:

    .. code-block:: none

        up y ^   x front (yaw=0)
             |  /
             | /
             |/
             0 ------> z right (yaw=-0.5*pi)


    the yaw is around the y axis, thus the rotation axis=1. The yaw is 0 at
    the positive direction of x axis, and decreases from the positive direction
    of x to the positive direction of z.

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The boxes
            data with shape (N, box_dim).
        box_dim (int): Number of the dimension of a box. Each row is
            (x_size, y_size, z_size, x, y, z, yaw). Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation. If False, the
            value of yaw will be set to 0 as minmax boxes. Defaults to True.
        origin (Tuple[float]): Relative position of the box origin.
            Defaults to (0.5, 0, 0.5). This will guide the box be converted
            to (0.5, 0, 0.5) mode.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """
    YAW_AXIS = 1

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: Tuple[float, float, float] = (0.5, 0, 0.5)
    ) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)

        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, box_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, \
            ('The box dimension must be 2 and the length of the last '
             f'dimension must be {box_dim}, but got boxes with shape '
             f'{tensor.shape}.')

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding 0 as
            # a fake yaw and set with_yaw to False
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw

        # turn (x_size, y_size, z_size, x, y, z, yaw) to (x, y, z, x_size, y_size, z_size, yaw)
        tensor = torch.cat((tensor[:, 3:6], tensor[:, 0:3], tensor[:, 6:7]), dim=1)
        self.tensor = tensor.clone()

        # TODO
        if origin != (0.5, 0, 0.5):
            dst = self.tensor.new_tensor((0.5, 0, 0.5))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def height(self) -> Tensor:
        """Tensor: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 4]

    @property
    def top_height(self) -> Tensor:
        """Tensor: A vector with top height of each box in shape (N, )."""
        # the positive direction is down rather than up
        return self.bottom_height + self.height

    @property
    def bottom_height(self) -> Tensor:
        """Tensor: A vector with bottom height of each box in shape (N, )."""
        return self.tensor[:, 1]

    # @property
    # def local_yaw(self) -> Tensor:
    #     """Tensor: A vector with local yaw of each box in shape (N, ).
    #     local_yaw equals to alpha in kitti, which is commonly used in monocular
    #     3D object detection task, so only :obj:`CameraInstance3DBoxes` has the
    #     property."""
    #     yaw = self.yaw
    #     loc = self.gravity_center
    #     local_yaw = yaw2local(yaw, loc)
    #
    #     return local_yaw

    @property
    def gravity_center(self) -> Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, [0, 2]] = bottom_center[:, [0, 2]] # xoz
        gravity_center[:, 1] = bottom_center[:, 1] + self.tensor[:, 4] * 0.5
        return gravity_center

    @property
    def corners(self) -> Tensor:
        """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
        x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        .. code-block:: none

                                           up y
                            front x           ^
                                 /            |
                                /             |
                  (x1, y1, z0) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y1, z0) + ----------- +   + (x1, y0, z1)
                            |  /      .   |  /
                            | / origin    | /
               (x0, y0, z0) + ----------- +------->left z
                                        (x0, y0, z1)

        Returns:
            Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 0, 0.5)
        corners_norm = corners_norm - dims.new_tensor([0.5, 0, 0.5])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        corners = rotation_3d_in_axis(
            corners, self.tensor[:, 6], axis=self.YAW_AXIS)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    @property
    def bev(self) -> Tensor:
        """Tensor: 2D BEV box of each box with rotation in XYWHR format, in
        shape (N, 5)."""
        bev = self.tensor[:, [0, 2, 3, 5, 6]].clone()
        return bev

    def rotate(
        self,
        angle: Union[Tensor, np.ndarray, float],
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray], Tuple[
            BasePoints, Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        if not isinstance(angle, Tensor):
            angle = self.tensor.new_tensor(angle)

        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, \
            f'invalid rotation angle shape {angle.shape}'

        if angle.numel() == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3],
                angle,
                axis=self.YAW_AXIS,
                return_mat=True)
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[2, 0]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 6] += angle

        if points is not None:
            if isinstance(points, Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    # def flip(
    #     self,
    #     bev_direction: str = 'horizontal',
    #     points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    # ) -> Union[Tensor, np.ndarray, BasePoints, None]:
    #     """Flip the boxes in BEV along given BEV direction.
    #
    #     In CAM coordinates, it flips the x (horizontal) or z (vertical) axis.
    #
    #     Args:
    #         bev_direction (str): Direction by which to flip. Can be chosen from
    #             'horizontal' and 'vertical'. Defaults to 'horizontal'.
    #         points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
    #             Points to flip. Defaults to None.
    #
    #     Returns:
    #         Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
    #         is None, the function returns None, otherwise it returns the
    #         flipped points.
    #     """
    #     assert bev_direction in ('horizontal', 'vertical')
    #     if bev_direction == 'horizontal':
    #         self.tensor[:, 0::7] = -self.tensor[:, 0::7]
    #         if self.with_yaw:
    #             self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
    #     elif bev_direction == 'vertical':
    #         self.tensor[:, 2::7] = -self.tensor[:, 2::7]
    #         if self.with_yaw:
    #             self.tensor[:, 6] = -self.tensor[:, 6]
    #
    #     if points is not None:
    #         assert isinstance(points, (Tensor, np.ndarray, BasePoints))
    #         if isinstance(points, (Tensor, np.ndarray)):
    #             if bev_direction == 'horizontal':
    #                 points[:, 0] = -points[:, 0]
    #             elif bev_direction == 'vertical':
    #                 points[:, 2] = -points[:, 2]
    #         elif isinstance(points, BasePoints):
    #             points.flip(bev_direction)
    #         return points

    @classmethod
    def height_overlaps(cls, boxes1: 'Threedfront3DBoxes',
                        boxes2: 'Threedfront3DBoxes') -> Tensor:
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`Threedfront3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`Threedfront3DBoxes`): Boxes 2 contain M boxes.

        Returns:
            Tensor: Calculated height overlap of the boxes.
        """
        assert isinstance(boxes1, Threedfront3DBoxes)
        assert isinstance(boxes2, Threedfront3DBoxes)

        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

        heighest_of_bottom = torch.max(boxes1_bottom_height,
                                       boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    # def convert_to(self,
    #                dst: int,
    #                rt_mat: Optional[Union[Tensor, np.ndarray]] = None,
    #                correct_yaw: bool = False) -> 'BaseInstance3DBoxes':
    #     """Convert self to ``dst`` mode.
    #
    #     Args:
    #         dst (int): The target Box mode.
    #         rt_mat (Tensor or np.ndarray, optional): The rotation and
    #             translation matrix between different coordinates.
    #             Defaults to None. The conversion from ``src`` coordinates to
    #             ``dst`` coordinates usually comes along the change of sensors,
    #             e.g., from camera to LiDAR. This requires a transformation
    #             matrix.
    #         correct_yaw (bool): Whether to convert the yaw angle to the target
    #             coordinate. Defaults to False.
    #
    #     Returns:
    #         :obj:`BaseInstance3DBoxes`: The converted box of the same type in
    #         the ``dst`` mode.
    #     """
    #     from .box_3d_mode import Box3DMode
    #
    #     # TODO: always set correct_yaw=True
    #     return Box3DMode.convert(
    #         box=self,
    #         src=Box3DMode.CAM,
    #         dst=dst,
    #         rt_mat=rt_mat,
    #         correct_yaw=correct_yaw)

    # def points_in_boxes_part(
    #         self,
    #         points: Tensor,
    #         boxes_override: Optional[Tensor] = None) -> Tensor:
    #     """Find the box in which each point is.
    #
    #     Args:
    #         points (Tensor): Points in shape (1, M, 3) or (M, 3), 3 dimensions
    #             are (x, y, z) in LiDAR or depth coordinate.
    #         boxes_override (Tensor, optional): Boxes to override `self.tensor`.
    #             Defaults to None.
    #
    #     Returns:
    #         Tensor: The index of the first box that each point is in with shape
    #         (M, ). Default value is -1 (if the point is not enclosed by any
    #         box).
    #     """
    #     from .coord_3d_mode import Coord3DMode
    #
    #     points_lidar = Coord3DMode.convert(points, Coord3DMode.CAM,
    #                                        Coord3DMode.LIDAR)
    #     if boxes_override is not None:
    #         boxes_lidar = boxes_override
    #     else:
    #         boxes_lidar = Coord3DMode.convert(
    #             self.tensor,
    #             Coord3DMode.CAM,
    #             Coord3DMode.LIDAR,
    #             is_point=False)
    #
    #     box_idx = super().points_in_boxes_part(points_lidar, boxes_lidar)
    #     return box_idx

    # def points_in_boxes_all(self,
    #                         points: Tensor,
    #                         boxes_override: Optional[Tensor] = None) -> Tensor:
    #     """Find all boxes in which each point is.
    #
    #     Args:
    #         points (Tensor): Points in shape (1, M, 3) or (M, 3), 3 dimensions
    #             are (x, y, z) in LiDAR or depth coordinate.
    #         boxes_override (Tensor, optional): Boxes to override `self.tensor`.
    #             Defaults to None.
    #
    #     Returns:
    #         Tensor: The index of all boxes in which each point is with shape
    #         (M, T).
    #     """
    #     from .coord_3d_mode import Coord3DMode
    #
    #     points_lidar = Coord3DMode.convert(points, Coord3DMode.CAM,
    #                                        Coord3DMode.LIDAR)
    #     if boxes_override is not None:
    #         boxes_lidar = boxes_override
    #     else:
    #         boxes_lidar = Coord3DMode.convert(
    #             self.tensor,
    #             Coord3DMode.CAM,
    #             Coord3DMode.LIDAR,
    #             is_point=False)
    #
    #     box_idx = super().points_in_boxes_all(points_lidar, boxes_lidar)
    #     return box_idx

def bbox_overlaps_3d(bboxes1, bboxes2, mode='iou'):
    """Calculate 3D IoU using cuda implementation.

    Note:
        This function calculates the IoU of 3D boxes based on their volumes.
        IoU calculator :class:`BboxOverlaps3D` uses this function to
        calculate the actual IoUs of boxes.

    Args:
        bboxes1 (torch.Tensor): with shape (N, 7+C),
            (x, y, z, x_size, y_size, z_size, ry, v*).
        bboxes2 (torch.Tensor): with shape (M, 7+C),
            (x, y, z, x_size, y_size, z_size, ry, v*).
        mode (str): "iou" (intersection over union) or
            iof (intersection over foreground).
        coordinate (str): 'camera' or 'lidar' coordinate system.

    Return:
        torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2
            with shape (M, N) (aligned mode is not supported currently).
    """
    assert bboxes1.size(-1) == bboxes2.size(-1) >= 7

    bboxes1 = Threedfront3DBoxes(bboxes1, box_dim=bboxes1.shape[-1])
    bboxes2 = Threedfront3DBoxes(bboxes2, box_dim=bboxes2.shape[-1])

    return bboxes1.overlaps(bboxes1, bboxes2, mode=mode)

def axis_aligned_bbox_overlaps_3d(bboxes1,
                                  bboxes2,
                                  mode='iou',
                                  is_aligned=False,
                                  eps=1e-6):
    """Calculate overlap between two set of axis aligned 3D bboxes. If
        ``is_aligned`` is ``False``, then calculate the overlaps between each bbox
        of bboxes1 and bboxes2, otherwise the overlaps between each aligned pair of
        bboxes1 and bboxes2.
        Args:
            bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
                format or empty.
            bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
                format or empty.
                B indicates the batch dim, in shape (B1, B2, ..., Bn).
                If ``is_aligned`` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or "giou" (generalized
                intersection over union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Defaults to False.
            eps (float, optional): A value added to the denominator for numerical
                stability. Defaults to 1e-6.
        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """

    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimension is 6
    assert (bboxes1.size(-1) == 6 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 6 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 3] -
             bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1]) * (
                 bboxes1[..., 5] - bboxes1[..., 2])
    area2 = (bboxes2[..., 3] -
             bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1]) * (
                 bboxes2[..., 5] - bboxes2[..., 2])

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3]
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(bboxes1[..., :, None, :3],
                       bboxes2[..., None, :, :3])  # [B, rows, cols, 3]
        rb = torch.min(bboxes1[..., :, None, 3:],
                       bboxes2[..., None, :, 3:])  # [B, rows, cols, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 3]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :3],
                                    bboxes2[..., None, :, :3])
            enclosed_rb = torch.max(bboxes1[..., :, None, 3:],
                                    bboxes2[..., None, :, 3:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

if __name__ == "__main__":
    import copy
    # l, h, w, x, y, z; y is up
    bboxs1 = torch.tensor([[2,2,2,1,0,1,0],[2*np.sqrt(2),2,2*np.sqrt(2),2,0,2,np.pi/4],[4,2,4,4,0,2,0]])
    bboxs2 = copy.deepcopy(bboxs1)
    iou3d = bbox_overlaps_3d(bboxs1,bboxs2)
    print(iou3d)

    # Bedroom-43072
    bboxs1 = torch.tensor([[2.70252, 2.5308, 0.48541199999999995, 2.436394495985243, 0.0, -0.4211000000000002, 0],
    [2.087582, 0.944233, 2.096882, 2.7704135208892557, 0.0, 1.2272080882320147, -1.5707872225948964],
    [0.3157180000000004, 0.534938, 0.42481999999999953, 2.3319708556191747, 0.0, 2.5425890455731373, 0],
    [0.6878459999999997, 1.1206645017, 0.5060780000000001, 1.0770355475531685, 0.0, 1.4641999999999995,1.5707872225948964],
    [0.47698399999999996, 0.4999555, 0.392196, 3.723230195866933, 0.0, 0.27623272885509564, -1.5707872225948964],
    [0.6203720000000001, 0.40628398999999993, 0.622566, 2.3177, 2.1927329788208008, 1.099, 0],
    [3.0185, 0.0, 3.4866, 2.3299800913347264, 0.0, 0.8684732302567336, 0]])
    bboxs1[:,3:6] -=  bboxs1[-1,3:6]
    bboxs2 = copy.deepcopy(bboxs1)
    iou3d = bbox_overlaps_3d(bboxs1, bboxs2)
    print(iou3d)

    # MasterBedroom-33296
    bboxs1 = torch.tensor([[1.0406779999999998, 1.098808, 0.4491219999999996, -5.370314764104733, 0.0, -4.066067038275899, 0],
                           [0.48633499999999996, 0.607177, 0.4845579999999998, -2.560559506120386, 0.0, -4.061041670654544, 0],
                           [0.5164280000000003, 0.459586, 0.5169589999999997, -5.209674750434523, 0.0, -3.495716691618656, -1.5707872225948964],
                           [2.10004, 1.5628199694824, 2.3316999999999997, -3.8389265345624257, 0.0, -3.056672030098832, 0],
                           [0.8127560000000003, 1.032781, 0.41430599999999984, -5.16852, 0.0, -1.3732266743977866, -3.141592653589793],
                           [0.8236240000000001, 0.807919, 0.823493, -4.06966, 1.8820679950714112, -2.61897, 0],
                           [2.962778, 2.39078, 0.6670359999999997, -2.153817233615451, 0.0, -2.706551157226562, -1.5707872225948964],
                           [5.82, 0.0, 3.0749999999999997, -3.4582392686804453, 0.0, -2.5683879173290936, 0]])
    scene_center = torch.tensor([-2.9265, 0.0, -2.7])
    bboxs1[:, 3:6] -= scene_center
    bboxs2 = copy.deepcopy(bboxs1)
    iou3d = bbox_overlaps_3d(bboxs1, bboxs2)
    print(iou3d)

