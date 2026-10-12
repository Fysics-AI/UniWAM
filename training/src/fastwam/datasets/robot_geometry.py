"""Camera-frame actions and image targets for the full UniWAM interface."""
import torch
import torch.nn.functional as F


def rotation_to_6d(matrix):
    """Column 0 followed by column 1 (not a row-major flattened 3x2 matrix)."""
    return matrix[..., :2].transpose(-1, -2).flatten(-2)


def rotation_from_6d(value):
    first, second = value[..., :3], value[..., 3:6]
    if (first.norm(dim=-1) < 1e-7).any():
        raise ValueError("Cannot decode a degenerate first rotation column.")
    x = F.normalize(first, dim=-1)
    residual = second - (x * second).sum(-1, keepdim=True) * x
    if (residual.norm(dim=-1) < 1e-7).any():
        raise ValueError("Cannot decode collinear rotation columns.")
    y = F.normalize(residual, dim=-1)
    return torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)


def poses_in_camera(position, rotation, camera_from_reference, reference_from_arm):
    """Transforms may broadcast over [H,2]. Camera pose is fixed at chunk start."""
    transform = camera_from_reference @ reference_from_arm
    rot, trans = transform[..., :3, :3], transform[..., :3, 3]
    return (rot @ position.unsqueeze(-1)).squeeze(-1) + trans, rot @ rotation


def project_points(position, intrinsics, width, height):
    pixels_h = (intrinsics @ position.unsqueeze(-1)).squeeze(-1)
    positive = position[..., 2] > 1e-6
    pixels = pixels_h[..., :2] / pixels_h[..., 2:].clamp_min(1e-6)
    valid = positive & (pixels[..., 0] >= 0) & (pixels[..., 0] < width)
    valid &= (pixels[..., 1] >= 0) & (pixels[..., 1] < height)
    return pixels / pixels.new_tensor([width, height]), valid


def manipulation_targets(position, rotation, gripper, arm_present, uv, quality):
    """Build [H,26]; quality: 0=invalid, 1=visible, 2=occluded.

    Position/rotation are in the current main-camera frame. UV labels use the
    per-frame main view, normalized to [0,1]. Appearance visibility is supplied
    by the labeler; geometric projection alone cannot determine occlusion.
    """
    if position.ndim != 3 or position.shape[1:] != (2, 3):
        raise ValueError("Position must be [H,2,3].")
    horizon = len(position)
    if rotation.shape != (horizon, 2, 3, 3) or gripper.shape != (horizon, 2):
        raise ValueError("Rotation/gripper shape mismatch.")
    if uv.shape != (horizon, 2, 2) or quality.shape != (horizon, 2):
        raise ValueError("UV/quality shape mismatch.")
    if not ((quality == 0) | (quality == 1) | (quality == 2)).all():
        raise ValueError("Quality must be 0, 1 or 2.")
    present = torch.as_tensor(arm_present, device=position.device, dtype=torch.bool)
    if present.shape != (2,) or not present.any():
        raise ValueError("arm_present must mark at least one of two arms.")
    physical = torch.cat((position, rotation_to_6d(rotation), gripper[..., None]), -1)
    physical_mask = present[None, :, None].expand(horizon, 2, 10)
    visible = (quality == 1) & present[None]
    label_valid = (quality != 0) & present[None]
    aux = torch.cat((uv, visible[..., None].to(uv)), -1)
    aux_mask = torch.cat((visible[..., None].expand(-1, -1, 2), label_valid[..., None]), -1)
    action = torch.cat((physical.flatten(1), aux.flatten(1)), -1)
    mask = torch.cat((physical_mask.flatten(1), aux_mask.flatten(1)), -1)
    if not torch.isfinite(action[mask]).all():
        raise ValueError("Supervised action/UV entries must be finite.")
    return torch.where(mask, action, 0), mask


def navigation_targets(base_poses, current_pose, bbox, map_uv, auxiliary_valid):
    """World SE(2) poses -> chunk-relative dx,dy,heading and six image targets."""
    delta = base_poses[:, :2] - current_pose[:2]
    angle = current_pose[2]
    c, s = angle.cos(), angle.sin()
    xy = torch.stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]), -1)
    heading = base_poses[:, 2] - angle
    heading = torch.atan2(heading.sin(), heading.cos())
    action = torch.cat((xy, heading[:, None], bbox, map_uv), -1)
    mask = torch.cat((torch.ones_like(action[:, :3], dtype=torch.bool), auxiliary_valid.bool()), -1)
    if mask.shape != action.shape or not torch.isfinite(action[mask]).all():
        raise ValueError("Navigation targets or validity mask are invalid.")
    return torch.where(mask, action, 0), mask


def camera_canvas(main, left_wrist=None, right_wrist=None):
    """[T,3,H,W] -> [3,T,320,384]; 256-pixel main panel over 64-pixel wrists."""
    if main.ndim != 4 or main.shape[1] != 3 or main.dtype != torch.uint8:
        raise ValueError("Main frames must be uint8 [T,3,H,W].")
    resize = lambda x, size: F.interpolate(x.float(), size=size, mode='bilinear', align_corners=False).round().byte()
    top = resize(main, (256, 384))
    panels = []
    for view in (left_wrist, right_wrist):
        if view is None:
            panels.append(torch.zeros((len(main), 3, 64, 192), dtype=torch.uint8, device=main.device))
        else:
            if view.ndim != 4 or view.shape[:2] != main.shape[:2] or view.dtype != torch.uint8:
                raise ValueError("Wrist frames must be synchronized uint8 [T,3,H,W].")
            panels.append(resize(view, (64, 192)))
    return torch.cat((top, torch.cat(panels, -1)), -2).permute(1, 0, 2, 3).contiguous()
