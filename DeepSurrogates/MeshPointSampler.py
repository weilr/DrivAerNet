import numpy as np
import torch


class MeshPointSampler:
    """
    提供多种3D点云采样方法的封装类，支持静态方法统一调用，便于数据增强或训练前降采样。
    使用方式：MeshPointSampler.sample(method_name, vertices, num_points)
    """

    def __init__(self):
        self.methods = {
            'random': self._random_sampling,
            'farthest': self._farthest_point_sampling,
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
