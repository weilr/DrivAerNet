import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


class MeshPointSampler:
    """
    提供多种3D点云采样方法的封装类，支持静态方法统一调用，便于数据增强或训练前降采样。
    使用方式：MeshPointSampler.sample(method_name, vertices, num_points)
    """

    def __init__(self):
        self.methods = {
            'random': self._random_sampling,
            'farthest': self._farthest_point_sampling,
            'uniform_grid': self._uniform_grid_sampling,
            'voxel_grid': self._voxel_grid_sampling,
            'fps_cluster': self._cluster_farthest_sampling,
        }

    def sample(self, method: str, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        根据选择的方法对点云进行采样。

        Args:
            method (str): 使用的采样方法名称。
            vertices (torch.Tensor): 输入点云 (N, 3)。
            num_points (int): 目标采样点数。

        Returns:
            torch.Tensor: 采样后的点云 (num_points, 3)。
        """
        if method not in self.methods:
            raise ValueError(f"不支持的采样方法：{method}")
        return self.methods[method](vertices, num_points)

    def _random_sampling(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        随机从点云中选择 num_points 个点。

        优点：速度快，易实现。
        缺点：采样可能不均匀，容易忽略边缘/细节区域。
        """
        indices = np.random.choice(vertices.shape[0], num_points, replace=False)
        return vertices[indices]

    def _farthest_point_sampling(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        最远点采样：每次选择距离已选点最远的点，直到达到目标数量。

        优点：点之间分布均匀，能代表空间结构。
        缺点：计算复杂度较高，速度较慢。
        """
        farthest_pts = torch.zeros((num_points, vertices.shape[1]), dtype=vertices.dtype)
        init_index = np.random.randint(vertices.shape[0])
        farthest_pts[0] = vertices[init_index]
        distances = torch.norm(vertices - vertices[init_index], dim=1)
        for i in range(1, num_points):
            idx = torch.argmax(distances)
            farthest_pts[i] = vertices[idx]
            distances = torch.minimum(distances, torch.norm(vertices - vertices[idx], dim=1))
        return farthest_pts

    def _uniform_grid_sampling(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        均匀网格采样：将点云划分成网格块，在每个块中取均值点作为代表。

        优点：均匀覆盖空间，保留全局结构。
        缺点：对稀疏区域表现较差，边界可能欠采样。
        """
        grid_size = int(np.ceil(np.cbrt(num_points)))
        min_vals = vertices.min(0)[0]
        max_vals = vertices.max(0)[0]
        grid_spacing = (max_vals - min_vals) / grid_size
        indices = ((vertices - min_vals) / grid_spacing).long()
        unique_idx, inverse_indices = torch.unique(indices, return_inverse=True, dim=0)
        centroids = torch.zeros((unique_idx.shape[0], vertices.shape[1]), dtype=vertices.dtype)
        for i in range(unique_idx.shape[0]):
            group = vertices[inverse_indices == i]
            centroids[i] = group.mean(0)
        if centroids.shape[0] >= num_points:
            selected = np.random.choice(centroids.shape[0], num_points, replace=False)
            return centroids[selected]
        else:
            pad = torch.zeros((num_points - centroids.shape[0], 3), dtype=vertices.dtype)
            return torch.cat((centroids, pad), dim=0)

    def _voxel_grid_sampling(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        体素格采样：将点云划分到立方体体素中，只保留每个体素的一个点。

        优点：简化点云，保留结构，速度快。
        缺点：随机选点可能导致局部结构丢失。
        """
        voxel_size = (vertices.max(0)[0] - vertices.min(0)[0]).max().item() / np.cbrt(num_points)
        coords = ((vertices - vertices.min(0)[0]) / voxel_size).floor()
        _, unique_indices = np.unique(coords.numpy(), axis=0, return_index=True)
        sampled = vertices[unique_indices]
        if sampled.shape[0] >= num_points:
            selected = np.random.choice(sampled.shape[0], num_points, replace=False)
            return sampled[selected]
        else:
            pad = torch.zeros((num_points - sampled.shape[0], 3), dtype=vertices.dtype)
            return torch.cat((sampled, pad), dim=0)

    def _cluster_farthest_sampling(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        基于局部密度 + 远点的聚类采样：
        - 首先估计每个点的局部密度（通过k近邻）。
        - 密度越高的点被采样概率越小，防止“堆积”。
        - 使用概率分布加权采样。

        优点：既考虑均匀性又保留稀疏区域特征。
        缺点：依赖kNN，计算成本略高。
        """
        np_vertices = vertices.numpy()
        nbrs = NearestNeighbors(n_neighbors=min(30, len(vertices))).fit(np_vertices)
        _, indices = nbrs.kneighbors(np_vertices)
        density = 1.0 / (np.std(np.diff(indices, axis=1), axis=1) + 1e-6)
        prob = density / density.sum()
        selected_indices = np.random.choice(len(vertices), size=num_points, replace=False, p=prob)
        return vertices[selected_indices]