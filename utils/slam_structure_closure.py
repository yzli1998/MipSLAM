import numpy as np
import open3d as o3d
import typing as Dict
import torch
from scipy.fft import fft, fftfreq
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from gaussian_splatting.utils.graphics_utils import getWorld2View2

from scipy.spatial.distance import cdist
from collections import deque


class StructureClosure:

    def __init__(self, config):
        super().__init__()

        print("Initializing Spectral-Aware Pose Graph Optimization (SA-PGO)...")
        self.pose_graph = None
        self.uid_to_node_id: Dict[int, int] = {}
        self.nodeid_to_uid: Dict[int, int] = {}
        self.first_uid = 0
        self.times = 0

        self.pose_trajectory = []
        self.pose_frequency_signature = {}

        self.spectral_gap = 1.0
        self.graph_connectivity_score = 1.0

        self.edge_spectral_weights = {}
        self.edge_frequency_coherence = {}
        self.edge_ages = {}
        self.previous_edge_weights = {}

        self.laplacian_eigenvecs = []

        self.monitor = Monitor()
        self.current_spectral_influence = 1.0
        self.conservative_mode = False
        self.degraded_mode = False

        self.use_spectral_analysis = False
        self.spectral_warmup_counter = 0
        self.hybrid_mode = False

        self.optimization_quality_history = deque(maxlen=20)

    def initialize_structure_graph(self, viewpoint):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()

        T_w2c = torch.eye(4, device=viewpoint.cam_trans_delta.device)
        T_w2c[:3, :3] = viewpoint.R.T
        T_w2c[:3, 3] = -viewpoint.R.T @ viewpoint.T

        node = o3d.pipelines.registration.PoseGraphNode(T_w2c.detach().cpu().numpy())

        self.first_uid = viewpoint.uid
        self.pose_graph.nodes.append(node)
        self.uid_to_node_id[viewpoint.uid] = len(self.pose_graph.nodes) - 1
        self.nodeid_to_uid[len(self.pose_graph.nodes) - 1] = viewpoint.uid

        pose_vec = self._pose_to_vector(T_w2c.detach().cpu().numpy())
        self.pose_trajectory.append(pose_vec)
        self.pose_frequency_signature[0] = np.zeros(6)

        self.monitor.initialize(pose_vec)

    def update_node(self, vp_j, keyframe_database):
        node_j = self.uid_to_node_id[vp_j.uid]

        T_w2c = torch.eye(4, device=vp_j.cam_trans_delta.device)
        T_w2c[:3, :3] = vp_j.R.T
        T_w2c[:3, 3] = -vp_j.R.T @ vp_j.T

        node = o3d.pipelines.registration.PoseGraphNode(T_w2c.detach().cpu().numpy())
        self.pose_graph.nodes[node_j] = node

        pose_vec = self._pose_to_vector(T_w2c.detach().cpu().numpy())
        if len(self.pose_trajectory) > node_j:
            self.pose_trajectory[node_j] = pose_vec
        else:
            self.pose_trajectory.append(pose_vec)

        self.monitor.update(pose_vec)

        if len(self.pose_trajectory) >= 15 and not self.degraded_mode:
            self.pose_frequency_signature[node_j] = self._compute_frequency_signature()
        else:
            self.pose_frequency_signature[node_j] = np.zeros(6)

        return True

    def add_viewpoint_to_graph(self, vp_j, keyframe_database, curr_window, voxel_size=0.01):
        T_w2c = torch.eye(4, device=vp_j.cam_trans_delta.device)
        T_w2c[:3, :3] = vp_j.R.T.float()
        T_w2c[:3, 3] = -vp_j.R.T.float() @ vp_j.T.float()

        node = o3d.pipelines.registration.PoseGraphNode(T_w2c.detach().cpu().numpy())

        self.pose_graph.nodes.append(node)
        node_id = len(self.pose_graph.nodes) - 1
        self.uid_to_node_id[vp_j.uid] = node_id
        self.nodeid_to_uid[node_id] = vp_j.uid

        if self.first_uid is None:
            self.first_uid = vp_j.uid

        pose_vec = self._pose_to_vector(T_w2c.detach().cpu().numpy())
        self.pose_trajectory.append(pose_vec)
        self.monitor.update(pose_vec)

        self._update_spectral_influence()
        self._check_and_update_modes()

        self.spectral_warmup_counter += 1
        if (self.spectral_warmup_counter >= 20 and
                not self.degraded_mode and
                not self.conservative_mode):
            self.use_spectral_analysis = True
            if node_id >= 100:
                self.hybrid_mode = True
            self.pose_frequency_signature[node_id] = self._compute_frequency_signature()
        else:
            self.pose_frequency_signature[node_id] = np.zeros(6)

        if node_id > 0:
            self._add_odometry_edge(vp_j, keyframe_database, node_id)

        self.times += 1
        self._run_optimization(vp_j)

    def _update_spectral_influence(self):
        self.current_spectral_influence *= 0.98
        self.current_spectral_influence = max(self.current_spectral_influence, 0.2)

        if len(self.pose_graph.nodes) > 100:
            late_stage_factor = 1.0 - (len(self.pose_graph.nodes) - 100) * 0.001
            late_stage_factor = max(late_stage_factor, 0.5)
            self.current_spectral_influence *= late_stage_factor

    def _check_and_update_modes(self):
        if self.monitor.is_degraded():
            if not self.degraded_mode:
                print("ALERT: Degradation detected, switching to conservative mode")
                self.degraded_mode = True
                self.conservative_mode = True
                self.use_spectral_analysis = False

        if len(self.pose_graph.nodes) > 100 and not self.conservative_mode:
            if not self.hybrid_mode:
                print(f"Entering hybrid mode at {len(self.pose_graph.nodes)} nodes")
                self.hybrid_mode = True

        if self.degraded_mode and self.monitor.is_recovered_for(50):
            print("Recovered, re-enabling conservative spectral analysis")
            self.degraded_mode = False
            self.conservative_mode = False
            self.use_spectral_analysis = True

    def _add_odometry_edge(self, vp_j, keyframe_database, node_j):
        node_i = node_j - 1
        vp_i = keyframe_database[self.nodeid_to_uid[node_i]]

        Ti = torch.eye(4, device=vp_i.cam_trans_delta.device)
        Tj = torch.eye(4, device=vp_j.cam_trans_delta.device)
        Ti[:3, :3], Ti[:3, 3] = vp_i.R.T.float(), -vp_i.R.T.float() @ vp_i.T.float()
        Tj[:3, :3], Tj[:3, 3] = vp_j.R.T.float(), -vp_j.R.T.float() @ vp_j.T.float()

        Tij = (torch.inverse(Ti) @ Tj).cpu().detach().numpy()

        if self.use_spectral_analysis and not self.conservative_mode and not self.degraded_mode:
            frequency_coherence = self._compute_frequency_coherence(node_i, node_j)
            spectral_confidence = self._compute_spectral_confidence(Tij, frequency_coherence)
            spectral_confidence = (spectral_confidence * self.current_spectral_influence +
                                   0.8 * (1.0 - self.current_spectral_influence))
        else:
            translation_norm = np.linalg.norm(Tij[:3, 3])
            rotation_angle = np.arccos(np.clip((np.trace(Tij[:3, :3]) - 1) / 2, -1, 1))
            spectral_confidence = np.exp(-translation_norm * 2.0 - rotation_angle * 4.0)
            spectral_confidence = np.clip(spectral_confidence, 0.4, 1.0)
            frequency_coherence = 0.8

        information_matrix = self._compute_information_matrix(
            Tij, spectral_confidence, frequency_coherence, 'odometry', node_j
        )

        edge = o3d.pipelines.registration.PoseGraphEdge(
            source_node_id=node_j,
            target_node_id=node_i,
            transformation=Tij,
            information=information_matrix,
            uncertain=False
        )

        edge_id = len(self.pose_graph.edges)
        self.pose_graph.edges.append(edge)

        self.edge_spectral_weights[edge_id] = spectral_confidence
        self.edge_frequency_coherence[edge_id] = frequency_coherence
        self.edge_ages[edge_id] = self.times
        self.previous_edge_weights[edge_id] = spectral_confidence

    def _run_optimization(self, vp_j):
        print(f"--------> Running SA-PGO Optimization (nodes: {len(self.pose_graph.nodes)}, "
              f"spectral_influence: {self.current_spectral_influence:.3f}) <---------")

        pre_optimization_error = self._evaluate_convergence()

        if (self.use_spectral_analysis and
                not self.conservative_mode and
                not self.degraded_mode and
                len(self.pose_graph.nodes) >= 15):
            self._update_graph_spectral_properties()

        graph_size = len(self.pose_graph.nodes)

        if self.degraded_mode or self.conservative_mode:
            edge_prune_threshold = 0.4
            max_correspondence = 0.06
            preference_loop_closure = 0.5
        elif graph_size > 100:
            edge_prune_threshold = 0.35
            max_correspondence = 0.05
            preference_loop_closure = 0.6
        elif self.use_spectral_analysis and self.spectral_gap < 0.05:
            edge_prune_threshold = 0.25
            max_correspondence = 0.04
            preference_loop_closure = 0.8
        else:
            edge_prune_threshold = 0.3
            max_correspondence = 0.05
            preference_loop_closure = 0.7

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence,
            edge_prune_threshold=edge_prune_threshold,
            preference_loop_closure=preference_loop_closure,
            reference_node=self.uid_to_node_id[self.first_uid],
        )

        if self.use_spectral_analysis and not self.conservative_mode:
            self._apply_edge_reweighting()

        o3d.pipelines.registration.global_optimization(
            self.pose_graph,
            method=o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            criteria=o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option=option
        )

        post_optimization_error = self._evaluate_convergence()
        optimization_quality = self._compute_optimization_quality(pre_optimization_error, post_optimization_error)

        self.optimization_quality_history.append(optimization_quality)
        self.monitor.update_optimization_result(optimization_quality, post_optimization_error)

        if len(self.optimization_quality_history) >= 5:
            recent_quality = np.mean(list(self.optimization_quality_history)[-5:])
            if recent_quality < 0.3:
                print("ALERT: Poor optimization quality detected")
                if not self.conservative_mode:
                    self.conservative_mode = True
                    self.use_spectral_analysis = False

        node_j = self.uid_to_node_id[vp_j.uid]
        optimized_T = self.pose_graph.nodes[node_j].pose

        vp_j.R = torch.tensor(optimized_T[:3, :3].T, device=vp_j.cam_trans_delta.device)
        vp_j.T = torch.tensor(-optimized_T[:3, :3].T @ optimized_T[:3, 3], device=vp_j.cam_trans_delta.device)

        if self.use_spectral_analysis:
            pose_vec = self._pose_to_vector(optimized_T)
            if node_j < len(self.pose_trajectory):
                self.pose_trajectory[node_j] = pose_vec

        mode_status = []
        if self.conservative_mode: mode_status.append("CONSERVATIVE")
        if self.hybrid_mode: mode_status.append("HYBRID")
        if self.degraded_mode: mode_status.append("DEGRADED")
        if self.use_spectral_analysis: mode_status.append("SPECTRAL")
        mode_str = "+".join(mode_status) if mode_status else "STANDARD"

        print(f"Mode: {mode_str}, Quality: {optimization_quality:.3f}, "
              f"Error: {pre_optimization_error:.4f} -> {post_optimization_error:.4f}")

        if self.use_spectral_analysis:
            print(f"Spectral: gap={self.spectral_gap:.4f}, connectivity={self.graph_connectivity_score:.4f}, "
                  f"influence={self.current_spectral_influence:.3f}")

    def match_viewpoint(self, vp_j, keyframe_database, threshold=0.75, voxel_size=0.01,
                        adjacent_range=35, loop_closure_interval=50, max_loop_candidates=4):
        spectral_candidates = []
        if (self.use_spectral_analysis and
                not self.conservative_mode and
                not self.degraded_mode and
                self.current_spectral_influence > 0.3):
            spectral_candidates = self._spectral_clustering_candidates(vp_j, keyframe_database)

        base_threshold = threshold
        if self.degraded_mode:
            adaptive_threshold = base_threshold + 0.15
        elif self.conservative_mode:
            adaptive_threshold = base_threshold + 0.1
        elif len(self.pose_graph.nodes) > 100:
            adaptive_threshold = base_threshold + 0.05
        else:
            adaptive_threshold = base_threshold

        if (self.use_spectral_analysis and
                not self.conservative_mode and
                self.uid_to_node_id[vp_j.uid] in self.pose_frequency_signature):
            frequency_signature = self.pose_frequency_signature[self.uid_to_node_id[vp_j.uid]]
            spectral_complexity = np.linalg.norm(frequency_signature[3:])
            spectral_adjustment = 0.02 * min(spectral_complexity, 0.2) * self.current_spectral_influence
            adaptive_threshold += spectral_adjustment

        current_desc_list = [vp_j.descriptor]
        ref_ids = []
        ref_descriptors = []
        match_types = []

        effective_adjacent_range = adjacent_range
        if self.conservative_mode or self.degraded_mode:
            effective_adjacent_range = min(20, adjacent_range)

        for kf_id, vp_i in keyframe_database.items():
            frame_diff = abs(kf_id - vp_j.uid)
            if 2 <= frame_diff <= effective_adjacent_range:
                if (self.use_spectral_analysis and
                        not self.conservative_mode and
                        kf_id in [self.nodeid_to_uid.get(nid, -1) for nid in spectral_candidates]):
                    ref_ids.append(kf_id)
                    ref_descriptors.append(vp_i.descriptor)
                    match_types.append('spectral_adjacent')
                elif frame_diff <= 10:
                    ref_ids.append(kf_id)
                    ref_descriptors.append(vp_i.descriptor)
                    match_types.append('adjacent')
                elif frame_diff <= 20 and not (self.conservative_mode or self.degraded_mode):
                    ref_ids.append(kf_id)
                    ref_descriptors.append(vp_i.descriptor)
                    match_types.append('adjacent')

        if vp_j.uid >= loop_closure_interval * 2 and not self.degraded_mode:
            loop_candidates = []

            if self.conservative_mode:
                recent_threshold = vp_j.uid - 60
                sample_rates = (15, 40)
            else:
                recent_threshold = vp_j.uid - 50
                sample_rates = (10, 30)

            for kf_id in keyframe_database.keys():
                if kf_id < vp_j.uid - loop_closure_interval:
                    if kf_id > recent_threshold:
                        if kf_id % sample_rates[0] == 0:
                            loop_candidates.append(kf_id)
                    else:
                        if kf_id % sample_rates[1] == 0:
                            loop_candidates.append(kf_id)

            if (self.use_spectral_analysis and
                    not self.conservative_mode and
                    not self.degraded_mode):
                for candidate_node_id in spectral_candidates:
                    candidate_uid = self.nodeid_to_uid.get(candidate_node_id)
                    if (candidate_uid and candidate_uid in keyframe_database and
                            candidate_uid < vp_j.uid - loop_closure_interval):
                        if candidate_uid not in loop_candidates:
                            loop_candidates.append(candidate_uid)

            max_candidates = max_loop_candidates
            if self.conservative_mode or self.degraded_mode:
                max_candidates = max(2, max_loop_candidates // 2)

            loop_candidates = sorted(loop_candidates, reverse=True)[:max_candidates]

            for kf_id in loop_candidates:
                if kf_id in keyframe_database:
                    ref_ids.append(kf_id)
                    ref_descriptors.append(keyframe_database[kf_id].descriptor)
                    if (self.use_spectral_analysis and
                            not self.conservative_mode and
                            kf_id in [self.nodeid_to_uid.get(nid, -1) for nid in spectral_candidates]):
                        match_types.append('spectral_loop')
                    else:
                        match_types.append('loop')

        if not ref_ids:
            return None

        desc1 = np.array(current_desc_list)
        desc2 = np.array(ref_descriptors)

        cosine_sim = 1 - cdist(desc1, desc2, metric='cosine')
        euclidean_dist = cdist(desc1, desc2, metric='euclidean')

        if np.max(euclidean_dist) > 0:
            euclidean_sim = 1 - (euclidean_dist / np.max(euclidean_dist))
        else:
            euclidean_sim = np.ones_like(euclidean_dist)

        if (self.use_spectral_analysis and
                not self.conservative_mode and
                not self.degraded_mode and
                self.current_spectral_influence > 0.3):
            spectral_weights = []
            for i, ref_id in enumerate(ref_ids):
                if ref_id in self.uid_to_node_id:
                    ref_node_id = self.uid_to_node_id[ref_id]
                    freq_coherence = self._compute_frequency_coherence(
                        self.uid_to_node_id[vp_j.uid], ref_node_id
                    )
                    weighted_coherence = (freq_coherence * self.current_spectral_influence +
                                          0.8 * (1.0 - self.current_spectral_influence))
                    spectral_weights.append(weighted_coherence)
                else:
                    spectral_weights.append(0.8)

            spectral_weights = np.array(spectral_weights).reshape(1, -1)
            spectral_weight_factor = min(0.15, 0.1 * self.current_spectral_influence)
            similarity_matrix = (0.7 * cosine_sim + 0.3 * euclidean_sim +
                                 spectral_weight_factor * spectral_weights)
        else:
            similarity_matrix = 0.75 * cosine_sim + 0.25 * euclidean_sim

        matches = []
        for i in range(len(desc1)):
            similarities = similarity_matrix[i]
            for idx, (similarity, match_type) in enumerate(zip(similarities, match_types)):
                effective_threshold = adaptive_threshold
                if 'spectral' in match_type and not self.conservative_mode:
                    spectral_bonus = 0.02 * self.current_spectral_influence
                    effective_threshold -= spectral_bonus
                elif match_type == 'loop':
                    effective_threshold += 0.12

                if similarity >= effective_threshold:
                    original_kf_id = ref_ids[idx]
                    matches.append((i, original_kf_id, similarity, match_type))

        if not matches:
            print("No matches found.")
            return None

        matches.sort(key=lambda x: (x[3] == 'adjacent', 'spectral' in x[3] and not self.conservative_mode, x[2]),
                     reverse=True)

        max_matches = 3
        if self.conservative_mode or self.degraded_mode:
            max_matches = 2
        elif len(self.pose_graph.nodes) > 100:
            max_matches = 2

        matches = matches[:max_matches]

        edges_created = 0
        for match in matches:
            match_idx = match[1]
            confidence = match[2]
            match_type = match[3]

            if self._create_match_edge(vp_j, keyframe_database, match_idx, confidence, match_type, voxel_size):
                edges_created += 1
                if edges_created >= max_matches:
                    break

        print(f"Matching: Created {edges_created} edges from {len(matches)} candidates")

        status_parts = []
        if self.conservative_mode: status_parts.append("CONSERVATIVE")
        if self.degraded_mode: status_parts.append("DEGRADED")
        if self.hybrid_mode: status_parts.append("HYBRID")
        if self.use_spectral_analysis: status_parts.append(f"SPECTRAL({self.current_spectral_influence:.2f})")
        if status_parts:
            print(f"Status: {'+'.join(status_parts)}")

        if len(self.pose_graph.nodes) % 10 == 0:
            self._create_anchor_edge(vp_j, keyframe_database[self.first_uid])

        return edges_created

    def _create_match_edge(self, vp_j, keyframe_database, match_idx, confidence, match_type, voxel_size):
        try:
            node_i = self.uid_to_node_id[match_idx]
            node_j = self.uid_to_node_id[vp_j.uid]

            if abs(node_j - node_i) <= 1:
                return False

            vp_i = keyframe_database[match_idx]

            frequency_coherence = 0.8
            if (self.use_spectral_analysis and
                    not self.conservative_mode and
                    not self.degraded_mode):
                frequency_coherence = self._compute_frequency_coherence(node_i, node_j)
                min_coherence = 0.3 + (1.0 - self.current_spectral_influence) * 0.2
                if frequency_coherence < min_coherence:
                    return False

            pcd_i = o3d.geometry.PointCloud()
            pcd_i.points = o3d.utility.Vector3dVector(vp_i.cam_points)
            pcd_j = o3d.geometry.PointCloud()
            pcd_j.points = o3d.utility.Vector3dVector(vp_j.cam_points)

            min_points = 200 if self.conservative_mode else 150
            if len(pcd_i.points) < min_points or len(pcd_j.points) < min_points:
                return False

            pcd_i_down = pcd_i.voxel_down_sample(voxel_size)
            pcd_j_down = pcd_j.voxel_down_sample(voxel_size)

            min_down_points = 100 if self.conservative_mode else 80
            if len(pcd_i_down.points) < min_down_points or len(pcd_j_down.points) < min_down_points:
                return False

            pcd_i_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 30))
            pcd_j_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 30))

            camera_location_i = getWorld2View2(vp_i.R, vp_i.T).detach().cpu().numpy()
            camera_location_j = getWorld2View2(vp_j.R, vp_j.T).detach().cpu().numpy()

            pcd_i_down.orient_normals_towards_camera_location(camera_location=camera_location_i[:3, 3])
            pcd_j_down.orient_normals_towards_camera_location(camera_location=camera_location_j[:3, 3])

            Ti2j = camera_location_i @ np.linalg.inv(camera_location_j)

            distance_threshold = 0.035
            if self.conservative_mode or self.degraded_mode:
                distance_threshold = 0.03

            reg_result = o3d.pipelines.registration.registration_icp(
                pcd_j_down, pcd_i_down, distance_threshold,
                Ti2j,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)
            )

            if self.conservative_mode or self.degraded_mode:
                min_fitness = 0.65
                max_rmse = 0.025
            elif 'spectral' in match_type:
                min_fitness = 0.55
                max_rmse = 0.035
            else:
                min_fitness = 0.6
                max_rmse = 0.03

            if reg_result.fitness < min_fitness or reg_result.inlier_rmse > max_rmse:
                return False

            if not self._geometric_consistency_check(reg_result.transformation, match_type):
                return False

            information_matrix = self._compute_information_matrix(
                reg_result.transformation, confidence, frequency_coherence, match_type, node_j
            )

            edge = o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=node_j,
                target_node_id=node_i,
                transformation=reg_result.transformation,
                information=information_matrix,
                uncertain=False
            )

            edge_id = len(self.pose_graph.edges)
            self.pose_graph.edges.append(edge)

            edge_quality = reg_result.fitness * confidence * frequency_coherence
            self.edge_spectral_weights[edge_id] = edge_quality
            self.edge_frequency_coherence[edge_id] = frequency_coherence
            self.edge_ages[edge_id] = self.times
            self.previous_edge_weights[edge_id] = edge_quality

            print(f"Created {match_type} edge {vp_j.uid}->{match_idx}: "
                  f"fitness={reg_result.fitness:.3f}, freq_coherence={frequency_coherence:.3f}, "
                  f"quality={edge_quality:.3f}")
            return True

        except Exception as e:
            print(f"Error creating edge {vp_j.uid}-{match_idx}: {e}")
            return False

    def _create_anchor_edge(self, vp_j, vp_i, voxel_size=0.01):
        try:
            node_i = self.uid_to_node_id[vp_i.uid]
            node_j = self.uid_to_node_id[vp_j.uid]

            if abs(node_j - node_i) <= 1:
                return False

            frequency_coherence = 0.8
            if (self.use_spectral_analysis and
                    not self.conservative_mode and
                    not self.degraded_mode):
                frequency_coherence = self._compute_frequency_coherence(node_i, node_j)
                if frequency_coherence < 0.4:
                    frequency_coherence = 0.6

            pcd_i = o3d.geometry.PointCloud()
            pcd_i.points = o3d.utility.Vector3dVector(vp_i.cam_points)
            pcd_j = o3d.geometry.PointCloud()
            pcd_j.points = o3d.utility.Vector3dVector(vp_j.cam_points)
            pcd_i_down = pcd_i.voxel_down_sample(voxel_size)
            pcd_j_down = pcd_j.voxel_down_sample(voxel_size)

            pcd_i_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 30))
            pcd_j_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 30))
            camera_location_i = getWorld2View2(vp_i.R, vp_i.T).detach().cpu().numpy()
            camera_location_j = getWorld2View2(vp_j.R, vp_j.T).detach().cpu().numpy()
            pcd_i_down.orient_normals_towards_camera_location(camera_location=camera_location_i[:3, 3])
            pcd_j_down.orient_normals_towards_camera_location(camera_location=camera_location_j[:3, 3])

            Ti2j = camera_location_i @ np.linalg.inv(camera_location_j)

            reg_result = o3d.pipelines.registration.registration_icp(
                pcd_j_down, pcd_i_down, 0.03,
                Ti2j,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
            )

            min_fitness = 0.75 if self.conservative_mode else 0.7
            max_rmse = 0.02 if self.conservative_mode else 0.025

            if reg_result.fitness < min_fitness or reg_result.inlier_rmse > max_rmse:
                return False

            information_matrix = self._compute_information_matrix(
                reg_result.transformation, 1.0, frequency_coherence, 'anchor', node_j
            )

            edge = o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=node_j,
                target_node_id=node_i,
                transformation=reg_result.transformation,
                information=information_matrix,
                uncertain=False
            )

            edge_id = len(self.pose_graph.edges)
            self.pose_graph.edges.append(edge)

            self.edge_spectral_weights[edge_id] = frequency_coherence
            self.edge_frequency_coherence[edge_id] = frequency_coherence
            self.previous_edge_weights[edge_id] = frequency_coherence

            return True

        except Exception as e:
            print(f"Error creating anchor edge {vp_j.uid}-{vp_i.uid}: {e}")
            return False

    def _pose_to_vector(self, pose_matrix):
        translation = pose_matrix[:3, 3]
        rotation_matrix = pose_matrix[:3, :3]

        U, s, Vt = np.linalg.svd(rotation_matrix)
        if np.linalg.det(U @ Vt) < 0:
            Vt[-1, :] *= -1
        rotation_matrix = U @ Vt

        rotation_vector = self._rotation_matrix_to_axis_angle(rotation_matrix)
        return np.concatenate([translation, rotation_vector])

    def _rotation_matrix_to_axis_angle(self, R):
        try:
            R = (R + R.T) / 2
            U, s, Vt = np.linalg.svd(R)
            R = U @ Vt

            trace = np.clip(np.trace(R), -1, 3)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

            if np.abs(angle) < 1e-4:
                return np.zeros(3)

            if np.abs(angle - np.pi) < 1e-4:
                diag = np.diag(R + np.eye(3))
                i = np.argmax(diag)
                axis = np.zeros(3)
                axis[i] = np.sqrt(max(0, diag[i] / 2))
                for j in range(3):
                    if i != j:
                        axis[j] = R[i, j] / (2 * axis[i] + 1e-8)
            else:
                axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-8:
                    axis = axis / axis_norm
                else:
                    return np.zeros(3)

            return axis * angle

        except Exception:
            return np.zeros(3)

    def _compute_frequency_signature(self):
        try:
            if len(self.pose_trajectory) < 10:
                return np.zeros(6)

            if self.degraded_mode or self.conservative_mode:
                window_size = min(15, len(self.pose_trajectory))
            else:
                window_size = min(25, len(self.pose_trajectory))

            recent_trajectory = np.array(self.pose_trajectory[-window_size:])
            if recent_trajectory.shape[0] < 8:
                return np.zeros(6)

            frequency_signature = np.zeros(6)
            for i in range(6):
                signal = recent_trajectory[:, i]
                signal_std = np.std(signal)
                if signal_std < 1e-6:
                    frequency_signature[i] = 0.0
                    continue

                if len(signal) > 5:
                    from scipy.ndimage import gaussian_filter1d
                    signal = gaussian_filter1d(signal, sigma=0.5)

                signal = signal - np.mean(signal)
                signal = signal * np.hanning(len(signal))

                n_fft = max(32, 2 ** int(np.ceil(np.log2(len(signal)))))
                fft_result = fft(signal, n=n_fft)
                power_spectrum = np.abs(fft_result) ** 2

                freqs = fftfreq(n_fft)
                positive_freqs = freqs[:len(freqs) // 2]
                positive_power = power_spectrum[:len(power_spectrum) // 2]

                if len(positive_power) > 0 and np.sum(positive_power) > 1e-10:
                    weighted_sum = np.sum(positive_freqs * positive_power)
                    total_power = np.sum(positive_power)
                    frequency_signature[i] = weighted_sum / (total_power + 1e-10)
                    frequency_signature[i] = np.clip(frequency_signature[i], 0, 0.3)

                    power_concentration = np.max(positive_power) / (np.sum(positive_power) + 1e-10)
                    if power_concentration > 0.8:
                        frequency_signature[i] *= 0.5

            frequency_signature *= self.current_spectral_influence
            return frequency_signature

        except Exception:
            return np.zeros(6)

    def _compute_frequency_coherence(self, node_i, node_j):
        try:
            if (node_i not in self.pose_frequency_signature or
                    node_j not in self.pose_frequency_signature):
                return 0.8

            sig_i = self.pose_frequency_signature[node_i]
            sig_j = self.pose_frequency_signature[node_j]

            norm_i = np.linalg.norm(sig_i)
            norm_j = np.linalg.norm(sig_j)
            if norm_i < 1e-8 or norm_j < 1e-8:
                return 0.8

            cosine_coherence = np.dot(sig_i, sig_j) / (norm_i * norm_j)
            cosine_coherence = (cosine_coherence + 1) / 2

            sig_i_norm = (sig_i - np.mean(sig_i)) / (np.std(sig_i) + 1e-8)
            sig_j_norm = (sig_j - np.mean(sig_j)) / (np.std(sig_j) + 1e-8)
            xcorr_coherence = np.abs(np.dot(sig_i_norm, sig_j_norm)) / len(sig_i)
            xcorr_coherence = np.clip(xcorr_coherence, 0, 1)

            freq_distance = np.linalg.norm(sig_i - sig_j) / (norm_i + norm_j + 1e-8)
            freq_coherence = np.exp(-freq_distance * 5.0)

            combined_coherence = (0.5 * cosine_coherence +
                                  0.3 * xcorr_coherence +
                                  0.2 * freq_coherence)

            if self.degraded_mode or self.conservative_mode:
                combined_coherence = 0.6 * combined_coherence + 0.4 * 0.8

            return np.clip(combined_coherence, 0.2, 1.0)

        except Exception:
            return 0.8

    def _compute_spectral_confidence(self, transformation, frequency_coherence):
        try:
            translation_norm = np.linalg.norm(transformation[:3, 3])
            rotation_angle = np.arccos(np.clip((np.trace(transformation[:3, :3]) - 1) / 2, -1, 1))

            motion_confidence = np.exp(-translation_norm * 1.0 - rotation_angle * 2.0)
            motion_confidence = np.clip(motion_confidence, 0.4, 1.0)

            base_confidence = 0.8 * motion_confidence + 0.2 * frequency_coherence

            if self.degraded_mode:
                base_confidence *= 0.7
            elif self.conservative_mode:
                base_confidence *= 0.85

            decayed_confidence = (base_confidence * self.current_spectral_influence +
                                  0.7 * (1.0 - self.current_spectral_influence))

            return np.clip(decayed_confidence, 0.4, 1.0)

        except Exception:
            return 0.7

    def _compute_information_matrix(self, transformation, confidence, frequency_coherence, edge_type, node_id):
        try:
            if edge_type == 'anchor':
                base_trans, base_rot = 80.0, 40.0
                if self.conservative_mode: base_trans, base_rot = 120.0, 60.0
            elif edge_type == 'odometry':
                base_trans, base_rot = 25.0, 12.0
                if self.conservative_mode: base_trans, base_rot = 35.0, 18.0
            elif 'spectral' in edge_type:
                base_trans, base_rot = 12.0, 6.0
                if self.conservative_mode: base_trans, base_rot = 8.0, 4.0
            else:
                base_trans, base_rot = 8.0, 4.0
                if self.conservative_mode: base_trans, base_rot = 6.0, 3.0

            if (self.use_spectral_analysis and
                    not self.conservative_mode and
                    not self.degraded_mode):
                effective_coherence = (frequency_coherence * self.current_spectral_influence +
                                       0.8 * (1.0 - self.current_spectral_influence))
                spectral_boost = 1.0 + 0.3 * effective_coherence
                noise_damping = max(0.85, effective_coherence)
            else:
                spectral_boost = 1.0
                noise_damping = 1.0

            translation_norm = np.linalg.norm(transformation[:3, 3])
            rotation_angle = np.arccos(np.clip((np.trace(transformation[:3, :3]) - 1) / 2, -1, 1))

            motion_factor = np.exp(-translation_norm * 0.3 - rotation_angle * 0.6)
            motion_factor = np.clip(motion_factor, 0.6, 1.3)

            graph_size_factor = 1.0 / (1.0 + len(self.pose_graph.nodes) / 500.0)
            graph_size_factor = max(graph_size_factor, 0.5)

            raw_trans_weight = base_trans * spectral_boost * motion_factor * confidence * noise_damping * graph_size_factor
            raw_rot_weight = base_rot * spectral_boost * motion_factor * confidence * noise_damping * graph_size_factor

            raw_trans_weight = min(raw_trans_weight, base_trans * 5.0)
            raw_rot_weight = min(raw_rot_weight, base_rot * 5.0)

            if node_id in self.previous_edge_weights and len(self.pose_graph.edges) > 0:
                prev_weight = self.previous_edge_weights.get(len(self.pose_graph.edges), 1.0)
                trans_weight = (0.85 * raw_trans_weight + 0.15 * prev_weight * base_trans)
                rot_weight = (0.85 * raw_rot_weight + 0.15 * prev_weight * base_rot)
            else:
                trans_weight = raw_trans_weight
                rot_weight = raw_rot_weight

            min_trans = 1.0 if self.conservative_mode else 0.5
            min_rot = 0.5 if self.conservative_mode else 0.25
            trans_weight = max(trans_weight, min_trans)
            rot_weight = max(rot_weight, min_rot)

            return np.diag([trans_weight, trans_weight, trans_weight, rot_weight, rot_weight, rot_weight])

        except Exception:
            if edge_type == 'anchor':
                return np.diag([30.0, 30.0, 30.0, 15.0, 15.0, 15.0])
            elif edge_type == 'odometry':
                return np.diag([15.0, 15.0, 15.0, 7.0, 7.0, 7.0])
            else:
                return np.diag([5.0, 5.0, 5.0, 2.5, 2.5, 2.5])

    def _update_graph_spectral_properties(self):
        if len(self.pose_graph.nodes) < 8:
            self.spectral_gap = 1.0
            self.graph_connectivity_score = 1.0
            return

        try:
            n_nodes = len(self.pose_graph.nodes)
            adjacency = np.zeros((n_nodes, n_nodes))

            for edge_id, edge in enumerate(self.pose_graph.edges):
                i, j = edge.source_node_id, edge.target_node_id
                if 0 <= i < n_nodes and 0 <= j < n_nodes:
                    weight = np.clip(self.edge_spectral_weights.get(edge_id, 0.8), 0.2, 1.5)
                    edge_age = self.times - self.edge_ages.get(edge_id, self.times)
                    weight *= np.exp(-edge_age * 0.001)
                    adjacency[i, j] = weight
                    adjacency[j, i] = weight

            degree = np.sum(adjacency, axis=1)
            disconnected_nodes = np.where(degree < 1e-6)[0]

            if len(disconnected_nodes) > 0:
                for i in disconnected_nodes:
                    neighbors = []
                    for offset in [1, -1, 2, -2]:
                        neighbor = i + offset
                        if 0 <= neighbor < n_nodes and neighbor != i:
                            neighbors.append(neighbor)
                        if len(neighbors) >= 2:
                            break
                    for neighbor in neighbors:
                        adjacency[i, neighbor] = 0.3
                        adjacency[neighbor, i] = 0.3
                degree = np.sum(adjacency, axis=1)

            laplacian = np.diag(degree) - adjacency

            degree_reg = degree + 1e-6
            degree_sqrt_inv = np.diag(1.0 / np.sqrt(degree_reg))
            normalized_laplacian = degree_sqrt_inv @ laplacian @ degree_sqrt_inv
            normalized_laplacian += 1e-8 * np.eye(n_nodes)

            k = min(8, n_nodes - 1)
            if k > 0:
                eigenvals, eigenvecs = eigsh(csr_matrix(normalized_laplacian), k=k, which='SM')

                sort_idx = np.argsort(eigenvals)
                eigenvals = eigenvals[sort_idx]
                self.laplacian_eigenvecs = eigenvecs[:, sort_idx]

                if len(eigenvals) >= 2:
                    raw_gap = eigenvals[1] - eigenvals[0]
                    if self.degraded_mode:
                        self.spectral_gap = max(0.01, raw_gap * 0.5)
                    elif self.conservative_mode:
                        self.spectral_gap = max(0.02, raw_gap * 0.7)
                    else:
                        self.spectral_gap = max(0.005, raw_gap)
                else:
                    self.spectral_gap = 0.1

                if len(eigenvals) >= 2:
                    raw_connectivity = eigenvals[1]
                    if self.degraded_mode:
                        self.graph_connectivity_score = max(0.01, raw_connectivity * 0.5)
                    elif self.conservative_mode:
                        self.graph_connectivity_score = max(0.02, raw_connectivity * 0.7)
                    else:
                        self.graph_connectivity_score = max(0.005, raw_connectivity)
                else:
                    self.graph_connectivity_score = 0.1

        except Exception as e:
            print(f"Spectral analysis fallback: {e}")
            self.spectral_gap = 0.1
            self.graph_connectivity_score = 0.1

    def _spectral_clustering_candidates(self, vp_j, keyframe_database):
        try:
            if (len(self.pose_graph.nodes) < 20 or
                    len(self.laplacian_eigenvecs) == 0 or
                    self.conservative_mode or
                    self.degraded_mode):
                return []

            n_clusters = min(4, len(self.pose_graph.nodes) // 12)
            if n_clusters < 2:
                return []

            current_node_id = self.uid_to_node_id[vp_j.uid]
            if current_node_id >= len(self.laplacian_eigenvecs):
                return []

            n_features = min(2, self.laplacian_eigenvecs.shape[1])
            features = self.laplacian_eigenvecs[:, :n_features]

            current_features = features[current_node_id]
            distances = []

            min_distance = max(8, len(self.pose_graph.nodes) // 20)

            for i in range(len(features)):
                if i != current_node_id and abs(i - current_node_id) > min_distance:
                    try:
                        dist = np.linalg.norm(features[i] - current_features)
                        if self.current_spectral_influence < 0.5:
                            dist *= (1.0 + (0.5 - self.current_spectral_influence))
                        distances.append((i, dist))
                    except:
                        continue

            if not distances:
                return []

            distances.sort(key=lambda x: x[1])
            max_candidates = min(4, max(2, len(distances) // 5))
            return [x[0] for x in distances[:max_candidates]]

        except Exception as e:
            print(f"Spectral clustering fallback: {e}")
            return []

    def _apply_edge_reweighting(self):
        try:
            if len(self.pose_graph.edges) == 0 or self.conservative_mode or self.degraded_mode:
                return

            for edge_id, edge in enumerate(self.pose_graph.edges):
                if edge_id in self.edge_frequency_coherence:
                    coherence = self.edge_frequency_coherence[edge_id]
                    effective_coherence = (coherence * self.current_spectral_influence +
                                           0.8 * (1.0 - self.current_spectral_influence))

                    if effective_coherence > 0.85:
                        boost_factor = 1.1
                    elif effective_coherence < 0.4:
                        boost_factor = 0.95
                    else:
                        boost_factor = 1.0

                    current_info = edge.information
                    target_info = current_info * boost_factor
                    edge.information = 0.8 * current_info + 0.2 * target_info

        except Exception as e:
            print(f"Edge reweighting fallback: {e}")

    def _geometric_consistency_check(self, transformation, match_type):
        try:
            translation_norm = np.linalg.norm(transformation[:3, 3])
            rotation_matrix = transformation[:3, :3]

            det = np.linalg.det(rotation_matrix)
            if abs(det - 1.0) > 0.1:
                return False

            orthogonality_error = np.linalg.norm(rotation_matrix @ rotation_matrix.T - np.eye(3))
            if orthogonality_error > 0.1:
                return False

            if self.degraded_mode or self.conservative_mode:
                max_translation = 5.0 if 'adjacent' in match_type else 15.0
                max_rotation = np.pi / 4 if 'adjacent' in match_type else np.pi / 2
            elif 'spectral' in match_type:
                max_translation = 10.0
                max_rotation = np.pi * 0.6
            elif match_type == 'adjacent' or match_type == 'spectral_adjacent':
                max_translation = 6.0
                max_rotation = np.pi / 3
            else:
                max_translation = 20.0
                max_rotation = np.pi * 0.8

            if translation_norm > max_translation:
                return False

            rotation_angle = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1, 1))
            if rotation_angle > max_rotation:
                return False

            return True

        except Exception:
            return False

    def _compute_optimization_quality(self, pre_error, post_error):
        try:
            if pre_error < 1e-8:
                return 1.0 if post_error < 1e-8 else 0.0

            improvement_ratio = (pre_error - post_error) / pre_error

            if improvement_ratio > 0.1 and post_error < 0.1:
                quality = 1.0
            elif improvement_ratio > 0.05 and post_error < 0.2:
                quality = 0.8
            elif improvement_ratio > 0.0 and post_error < 0.5:
                quality = 0.6
            elif improvement_ratio > -0.1:
                quality = 0.4
            else:
                quality = 0.2

            return np.clip(quality, 0.0, 1.0)

        except Exception:
            return 0.5

    def _evaluate_convergence(self):
        try:
            if len(self.pose_graph.edges) == 0:
                return 0.0

            total_error = 0.0
            valid_edges = 0
            max_single_error = 0.0

            for edge in self.pose_graph.edges:
                try:
                    T_i = self.pose_graph.nodes[edge.source_node_id].pose
                    T_j = self.pose_graph.nodes[edge.target_node_id].pose
                    T_ij_measured = edge.transformation
                    T_ij_estimated = np.linalg.inv(T_i) @ T_j

                    error_T = np.linalg.inv(T_ij_measured) @ T_ij_estimated
                    trans_error = np.linalg.norm(error_T[:3, 3])
                    rot_error = np.arccos(np.clip((np.trace(error_T[:3, :3]) - 1) / 2, -1, 1))

                    edge_error = trans_error + rot_error
                    total_error += edge_error
                    max_single_error = max(max_single_error, edge_error)
                    valid_edges += 1
                except:
                    continue

            if valid_edges == 0:
                return 0.0

            avg_error = total_error / valid_edges
            if max_single_error > avg_error * 5.0:
                avg_error *= 1.5

            return avg_error

        except Exception:
            return 1.0


class Monitor:

    def __init__(self):
        self.pose_history = deque(maxlen=100)
        self.error_history = deque(maxlen=50)
        self.optimization_quality_history = deque(maxlen=30)
        self.score = 1.0
        self.consecutive_good = 0
        self.consecutive_bad = 0

    def initialize(self, initial_pose):
        self.pose_history.append(initial_pose)

    def update(self, pose_vector):
        self.pose_history.append(pose_vector)
        self._update_score()

    def update_optimization_result(self, quality, error):
        self.optimization_quality_history.append(quality)
        self.error_history.append(error)
        self._update_score()

    def _update_score(self):
        scores = []

        if len(self.pose_history) >= 10:
            recent_poses = np.array(list(self.pose_history)[-10:])
            pose_variations = np.std(recent_poses, axis=0)
            scores.append(np.exp(-np.mean(pose_variations) * 2.0))

        if len(self.optimization_quality_history) >= 5:
            recent_qualities = list(self.optimization_quality_history)[-5:]
            scores.append(np.exp(-np.std(recent_qualities) * 3.0))

        if len(self.error_history) >= 5:
            recent_errors = list(self.error_history)[-5:]
            error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
            scores.append(np.exp(-abs(error_trend) * 10.0))

        if scores:
            self.score = np.mean(scores)

        if self.score < 0.3:
            self.consecutive_bad += 1
            self.consecutive_good = 0
        elif self.score > 0.7:
            self.consecutive_good += 1
            self.consecutive_bad = 0
        else:
            self.consecutive_bad = 0
            self.consecutive_good = 0

    def is_degraded(self):
        return self.score < 0.3 and self.consecutive_bad >= 3

    def is_recovered_for(self, periods):
        return self.consecutive_good >= periods