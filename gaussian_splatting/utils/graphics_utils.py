#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
import cv2
import open3d as o3d

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = translate.to(R.device)
    Rt = torch.zeros((4, 4), device=R.device)
    # Rt[:3, :3] = R.transpose()
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * (zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def fusion_point(pivot_kf, cur_kf, voxel_size=0.01):
    vp_j = cur_kf
    vp_i = pivot_kf

    pcd_i = o3d.geometry.PointCloud()
    pcd_i.points = o3d.utility.Vector3dVector(vp_i.cam_points)

    pcd_j = o3d.geometry.PointCloud()
    pcd_j.points = o3d.utility.Vector3dVector(vp_j.cam_points)

    pcd_i_down = pcd_i.voxel_down_sample(voxel_size)
    pcd_j_down = pcd_j.voxel_down_sample(voxel_size)

    pcd_i_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 30))
    camera_location_i = getWorld2View2(vp_i.R, vp_i.T).detach().cpu().numpy()
    pcd_i_down.orient_normals_towards_camera_location(camera_location=camera_location_i[:3, 3])

    pcd_j_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 30))
    camera_location_j = getWorld2View2(vp_j.R, vp_j.T).detach().cpu().numpy()
    pcd_j_down.orient_normals_towards_camera_location(camera_location=camera_location_j[:3, 3])

    Ti2j = camera_location_i @ np.linalg.inv(camera_location_j)

    icp_coarse = o3d.pipelines.registration.registration_icp(
        pcd_j_down, pcd_i_down, 0.05, Ti2j,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    trans_err, rot_err = transformation_updates(Ti2j, icp_coarse.transformation)
    if trans_err > 0.1 or rot_err > 10:
        print(f"⚠️  ICP deviated too far from initial guess: {trans_err:.3f} m, {rot_err:.1f}°")
        return None, None, None

    if icp_coarse.fitness < 0.3 or icp_coarse.inlier_rmse > 0.1:
        print("⚠️  ICP result may be unreliable:", icp_coarse.fitness, icp_coarse.inlier_rmse)
        return None, None, None

    icp_fine = o3d.pipelines.registration.registration_icp(
        pcd_j_down, pcd_i_down, 0.03, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    if icp_fine.inlier_rmse > 0.02:
        print("⚠️  refine ICP result may be unreliable:", icp_fine.fitness, icp_fine.inlier_rmse)
        return None, None, None

    pcd_j_in_pivot = pcd_j_down.transform(icp_fine.transformation)
    merged_pcd = pcd_j_in_pivot + pcd_i_down
    merged_pcd_down = merged_pcd.voxel_down_sample(voxel_size)
    vp_i.cam_points = np.asarray(merged_pcd_down.points)


def transformation_updates(T1, T2):
    # Compute relative transformation: ΔT = T1⁻¹ @ T2
    delta_T = np.linalg.inv(T1) @ T2

    # Translational difference (in meters)
    trans_error = np.linalg.norm(delta_T[:3, 3])

    # Rotational difference (in degrees)
    R = delta_T[:3, :3]
    angle_rad = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    rot_error = np.degrees(angle_rad)

    return trans_error, rot_error

def extract_descriptor(image: np.ndarray, descriptor_size: int = 128) -> np.ndarray:
    """
    优化的混合描述子提取，使用CPU上的numpy运算
   
    Args:
        image: 输入图像，形状为 (H, W, 3) 或 (3, H, W)，取值范围 [0, 1] 或 [0, 255]
        descriptor_size: 描述子维度，默认128
   
    Returns:
        描述子向量，numpy数组，形状为 (descriptor_size,)
    """
    # 输入格式处理和归一化
    if len(image.shape) == 3:
        if image.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
            image = np.transpose(image, (1, 2, 0))
        H, W, C = image.shape
    else:
        raise ValueError("图像格式不支持")
    
    # 归一化到 [0, 1]
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    
    # 转换为灰度图像
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    # 特征维度分配
    freq_size = descriptor_size // 2      # 64维 - 频域特征
    gradient_size = descriptor_size // 4  # 32维 - 梯度特征
    texture_size = descriptor_size // 8   # 16维 - 纹理特征
    color_size = descriptor_size - freq_size - gradient_size - texture_size  # 16维 - 颜色特征
    
    all_features = []
    
    # ===== 1. 频域特征提取 =====
    # 自适应预处理
    kernel_size = max(3, min(H, W) // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    gray_enhanced = gray - 0.3 * local_mean
    
    # 2D FFT
    fft_complex = np.fft.fft2(gray_enhanced)
    fft_shifted = np.fft.fftshift(fft_complex)
    
    # 幅度谱和相位谱
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)
    log_magnitude = np.log1p(magnitude)
    
    # 极坐标系
    center_h, center_w = H // 2, W // 2
    y, x = np.mgrid[0:H, 0:W]
    y_c, x_c = y.astype(np.float32) - center_h, x.astype(np.float32) - center_w
    radius = np.sqrt(x_c**2 + y_c**2)
    angle = np.arctan2(y_c, x_c + 1e-8)
    max_radius = min(center_h, center_w) * 0.9
    
    freq_features = []
    
    # 径向频率特征
    num_rings = max(1, freq_size // 4)
    for i in range(num_rings):
        r_inner = (i * max_radius) / num_rings
        r_outer = ((i + 1) * max_radius) / num_rings
        ring_mask = (radius >= r_inner) & (radius < r_outer)
        
        if np.sum(ring_mask) > 0:
            ring_mag_mean = np.mean(log_magnitude[ring_mask])
            ring_mag_std = np.std(log_magnitude[ring_mask])
            freq_features.extend([ring_mag_mean, ring_mag_std])
        else:
            freq_features.extend([0.0, 0.0])
    
    # 方向性频率特征
    num_sectors = max(1, freq_size // 4)
    angle_step = 2 * np.pi / num_sectors
    
    for i in range(num_sectors):
        angle_start = i * angle_step - np.pi
        angle_end = (i + 1) * angle_step - np.pi
        
        if angle_end <= np.pi:
            sector_mask = (angle >= angle_start) & (angle < angle_end) & (radius > max_radius * 0.1)
        else:
            sector_mask = ((angle >= angle_start) | (angle < angle_end - 2*np.pi)) & (radius > max_radius * 0.1)
        
        if np.sum(sector_mask) > 0:
            sector_energy = np.mean(log_magnitude[sector_mask])
            freq_features.append(sector_energy)
        else:
            freq_features.append(0.0)
    
    # 相位特征
    num_phase = max(1, freq_size // 4)
    for i in range(num_phase):
        r_inner = (i * max_radius) / num_phase
        r_outer = ((i + 1) * max_radius) / num_phase
        ring_mask = (radius >= r_inner) & (radius < r_outer)
        
        if np.sum(ring_mask) > 0:
            ring_phase = phase[ring_mask]
            phase_real = np.cos(ring_phase)
            phase_imag = np.sin(ring_phase)
            phase_consistency = np.sqrt(np.mean(phase_real)**2 + np.mean(phase_imag)**2)
            freq_features.append(phase_consistency)
        else:
            freq_features.append(0.0)
    
    # 频域纹理特征
    texture_freq_num = max(1, freq_size // 4)
    grid_size = max(1, int(np.sqrt(texture_freq_num)))
    block_h = max(1, H // grid_size)
    block_w = max(1, W // grid_size)
    
    for i in range(grid_size):
        for j in range(grid_size):
            start_h = i * block_h
            end_h = min((i + 1) * block_h, H)
            start_w = j * block_w
            end_w = min((j + 1) * block_w, W)
            
            block = log_magnitude[start_h:end_h, start_w:end_w]
            if block.size > 0:
                block_std = np.std(block)
                freq_features.append(block_std)
            else:
                freq_features.append(0.0)
            
            if len(freq_features) >= freq_size:
                break
        if len(freq_features) >= freq_size:
            break
    
    # 确保频域特征数量正确
    freq_features = freq_features[:freq_size] if len(freq_features) > freq_size else freq_features + [0.0] * (freq_size - len(freq_features))
    all_features.extend(freq_features)
    
    # ===== 2. 梯度特征提取 =====
    # Sobel算子
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude_grad = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    # 网格化统计
    grid_size = max(1, int(np.sqrt(gradient_size // 2)))
    cell_h = max(1, H // grid_size)
    cell_w = max(1, W // grid_size)
    
    gradient_features = []
    for i in range(grid_size):
        for j in range(grid_size):
            h_start = i * cell_h
            h_end = min((i + 1) * cell_h, H)
            w_start = j * cell_w
            w_end = min((j + 1) * cell_w, W)
            
            cell_mag = magnitude_grad[h_start:h_end, w_start:w_end]
            if cell_mag.size > 0:
                gradient_features.extend([np.mean(cell_mag), np.std(cell_mag)])
            else:
                gradient_features.extend([0.0, 0.0])
            
            if len(gradient_features) >= gradient_size:
                break
        if len(gradient_features) >= gradient_size:
            break
    
    # 确保梯度特征数量正确
    gradient_features = gradient_features[:gradient_size] if len(gradient_features) > gradient_size else gradient_features + [0.0] * (gradient_size - len(gradient_features))
    all_features.extend(gradient_features)
    
    # ===== 3. 纹理特征提取 =====
    # 多个滤波器
    kernel1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)  # Laplacian
    kernel2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)   # LoG approximation
    kernel3 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)  # Edge detector
    
    filters = [kernel1, kernel2, kernel3]
    features_per_filter = max(1, texture_size // len(filters))
    
    texture_features = []
    for filt in filters:
        filtered = cv2.filter2D(gray, cv2.CV_32F, filt)
        
        # 分区域统计
        grid_size = max(1, int(np.sqrt(features_per_filter // 2)))
        region_h = max(1, H // grid_size)
        region_w = max(1, W // grid_size)
        
        for i in range(grid_size):
            for j in range(grid_size):
                h_start = i * region_h
                h_end = min((i + 1) * region_h, H)
                w_start = j * region_w
                w_end = min((j + 1) * region_w, W)
                
                region = filtered[h_start:h_end, w_start:w_end]
                if region.size > 0:
                    texture_features.extend([np.mean(np.abs(region)), np.std(region)])
                else:
                    texture_features.extend([0.0, 0.0])
                
                if len(texture_features) >= texture_size:
                    break
            if len(texture_features) >= texture_size:
                break
        if len(texture_features) >= texture_size:
            break
    
    # 确保纹理特征数量正确
    texture_features = texture_features[:texture_size] if len(texture_features) > texture_size else texture_features + [0.0] * (texture_size - len(texture_features))
    all_features.extend(texture_features)
    
    # ===== 4. 颜色特征提取 =====
    bins_per_channel = max(1, color_size // 3)
    color_features = []
    
    for channel in range(3):
        channel_data = image[:, :, channel].flatten()
        if channel_data.size > 0:
            hist, _ = np.histogram(channel_data, bins=bins_per_channel, range=(0, 1))
            hist_norm = hist / (np.sum(hist) + 1e-8)
            color_features.extend(hist_norm)
        else:
            color_features.extend([0.0] * bins_per_channel)
    
    # 确保颜色特征数量正确
    color_features = color_features[:color_size] if len(color_features) > color_size else color_features + [0.0] * (color_size - len(color_features))
    all_features.extend(color_features)
    
    # ===== 5. 最终处理 =====
    descriptor = np.array(all_features, dtype=np.float32)
    
    # 确保维度正确
    if len(descriptor) > descriptor_size:
        descriptor = descriptor[:descriptor_size]
    elif len(descriptor) < descriptor_size:
        descriptor = np.concatenate([descriptor, np.zeros(descriptor_size - len(descriptor), dtype=np.float32)])
    
    # 增强归一化
    descriptor = np.maximum(descriptor, 0)  # ReLU激活
    
    # 分段归一化
    seg_size = max(1, len(descriptor) // 4)
    normalized_segments = []
    
    for i in range(4):
        start_idx = i * seg_size
        end_idx = (i + 1) * seg_size if i < 3 else len(descriptor)
        segment = descriptor[start_idx:end_idx]
        
        if segment.size > 0:
            seg_norm = np.linalg.norm(segment)
            if seg_norm > 1e-8:
                segment_normalized = segment / seg_norm
            else:
                segment_normalized = segment
        else:
            segment_normalized = np.zeros_like(segment)
        
        normalized_segments.append(segment_normalized)
    
    descriptor_normalized = np.concatenate(normalized_segments)
    
    # 最终L2归一化
    final_norm = np.linalg.norm(descriptor_normalized)
    if final_norm > 1e-8:
        descriptor_normalized = descriptor_normalized / final_norm
    
    return descriptor_normalized