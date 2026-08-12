import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
import trimesh
import yaml
import tqdm
import logging
from argparse import ArgumentParser
import importlib
from scipy.interpolate import Rbf

import smplx
from network.volume import CanoBlendWeightVolume
from utils.renderer import Renderer
import config
import pyexr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_pos_map(pos_map, path):
    """Save position map as a point cloud."""
    mask = np.linalg.norm(pos_map, axis=-1) > 0.
    positions = pos_map[mask]
    logging.info(f'Point nums {positions.shape[0]}')
    pc = trimesh.PointCloud(positions)
    pc.export(path)

def rbf_interpolate_lbs(pts, vertices, vertex_lbs):
    """
    Interpolate LBS weights using Radial Basis Function (RBF).
    :param pts: (N, 3) Points to interpolate weights for.
    :param vertices: (M, 3) SMPL vertices.
    :param vertex_lbs: (M, K) LBS weights for SMPL vertices.
    :return: (N, K) Interpolated weights for the points.
    """
    # Create RBF interpolators for each weight component
    rbf_interpolators = [
        Rbf(vertices[:, 0], vertices[:, 1], vertices[:, 2], vertex_lbs[:, k], function='thin_plate')
        for k in range(vertex_lbs.shape[1])
    ]
    
    # Interpolate weights for the points
    pts_lbs = np.stack([rbf_interpolators[k](pts[:, 0], pts[:, 1], pts[:, 2]) for k in range(vertex_lbs.shape[1])], axis=-1)
    return pts_lbs

def render_position_maps(renderer, vertices, normals, front_mv, back_mv):
    """Render position and normal maps for front and back views."""
    renderer.set_model(vertices, vertices)
    renderer.set_camera(front_mv)
    front_pos_map = renderer.render()[:, :, :3]

    renderer.set_camera(back_mv)
    back_pos_map = renderer.render()[:, :, :3]
    back_pos_map = cv.flip(back_pos_map, 1)
    pos_map = np.concatenate([front_pos_map, back_pos_map], 1)

    renderer.set_model(vertices, normals)
    renderer.set_camera(front_mv)
    front_nml_map = renderer.render()[:, :, :3]

    renderer.set_camera(back_mv)
    back_nml_map = renderer.render()[:, :, :3]
    back_nml_map = cv.flip(back_nml_map, 1)
    nml_map = np.concatenate([front_nml_map, back_nml_map], 1)

    return pos_map, nml_map

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, encoding='UTF-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type=str, help='Configuration file path.')
    args = arg_parser.parse_args()

    opt = load_config(args.config_path)
    dataset_module = opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
    MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
    dataset = MvRgbDataset(**opt['train']['data'])
    data_dir, frame_list = dataset.data_dir, dataset.pose_list
    logging.info(f"Frame list is {frame_list}")
    os.makedirs(data_dir + '/smpl_pos_map', exist_ok=True)

    cano_renderer = Renderer(1024, 1024, shader_name='vertex_attribute')

    smpl_model = smplx.SMPLX(config.PROJ_DIR + '/smpl_files/smplx', gender='neutral', use_pca=False, num_pca_comps=45, flat_hand_mean=True, batch_size=1)
    smpl_data = np.load(data_dir + '/smpl_params.npz')
    smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}

    with torch.no_grad():
        cano_smpl = smpl_model.forward(
            betas=smpl_data['betas'],
            global_orient=config.cano_smpl_global_orient[None],
            transl=config.cano_smpl_transl[None],
            body_pose=config.cano_smpl_body_pose[None]
        )
        cano_smpl_v = cano_smpl.vertices[0].cpu().numpy()
        cano_center = 0.5 * (cano_smpl_v.min(0) + cano_smpl_v.max(0))
        smpl_faces = smpl_model.faces.astype(np.int64)

    if os.path.exists(data_dir + '/template.ply'):
        logging.info(f'Loading template from {data_dir}/template.ply')
        template = trimesh.load(data_dir + '/template.ply', process=False)
        using_template = True
    else:
        logging.info(f'Cannot find template.ply from {data_dir}, using SMPL-X as template')
        template = trimesh.Trimesh(cano_smpl_v, smpl_faces, process=False)
        using_template = False

    cano_smpl_v = template.vertices.astype(np.float32)
    smpl_faces = template.faces.astype(np.int64)
    cano_smpl_v_dup = cano_smpl_v[smpl_faces.reshape(-1)]
    cano_smpl_n_dup = template.vertex_normals.astype(np.float32)[smpl_faces.reshape(-1)]

    # Define front & back view matrices
    front_mv = np.identity(4, np.float32)
    front_mv[:3, 3] = -cano_center + np.array([0, 0, -10], np.float32)
    front_mv[1:3] *= -1

    back_mv = np.identity(4, np.float32)
    rot_y = cv.Rodrigues(np.array([0, np.pi, 0], np.float32))[0]
    back_mv[:3, :3] = rot_y
    back_mv[:3, 3] = -rot_y @ cano_center + np.array([0, 0, -10], np.float32)
    back_mv[1:3] *= -1

    # Render canonical SMPL position and normal maps
    cano_pos_map, cano_nml_map = render_position_maps(cano_renderer, cano_smpl_v_dup, cano_smpl_n_dup, front_mv, back_mv)
    pyexr.write(data_dir + '/smpl_pos_map/cano_smpl_pos_map.exr', cano_pos_map)
    pyexr.write(data_dir + '/smpl_pos_map/cano_smpl_nml_map.exr', cano_nml_map)

    body_mask = np.linalg.norm(cano_pos_map, axis=-1) > 0.
    cano_pts = cano_pos_map[body_mask]
    if using_template:
        weight_volume = CanoBlendWeightVolume(data_dir + '/cano_weight_volume.npz')
        pts_lbs = weight_volume.forward_weight(torch.from_numpy(cano_pts)[None].cuda())[0]
    else:
        pts_lbs = rbf_interpolate_lbs(cano_pts, cano_smpl_v, smpl_model.lbs_weights.cpu().numpy())
        pts_lbs = torch.from_numpy(pts_lbs).cuda()
    np.save(data_dir + '/smpl_pos_map/init_pts_lbs.npy', pts_lbs.cpu().numpy())

    inv_cano_smpl_A = torch.linalg.inv(cano_smpl.A).cuda()
    body_mask = torch.from_numpy(body_mask).cuda()
    cano_pts = torch.from_numpy(cano_pts).cuda()
    pts_lbs = pts_lbs.cuda()

    for pose_idx in tqdm.tqdm(frame_list, desc='Generating positional maps...'):
        with torch.no_grad():
            live_smpl_woRoot = smpl_model.forward(
                betas=smpl_data['betas'],
                body_pose=smpl_data['body_pose'][pose_idx][None],
                jaw_pose=smpl_data['jaw_pose'][pose_idx][None],
                expression=smpl_data['expression'][pose_idx][None],
            )

        cano2live_jnt_mats_woRoot = torch.matmul(live_smpl_woRoot.A.cuda(), inv_cano_smpl_A)[0]
        pt_mats = torch.einsum('nj,jxy->nxy', pts_lbs, cano2live_jnt_mats_woRoot)
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros((1024, 2 * 1024, 3)).to(live_pts)
        live_pos_map[body_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode='nearest')[0]
        live_pos_map = live_pos_map.permute(1, 2, 0).cpu().numpy()

        pyexr.write(data_dir + '/smpl_pos_map/%08d.exr' % pose_idx, live_pos_map)

if __name__ == '__main__':
    main()