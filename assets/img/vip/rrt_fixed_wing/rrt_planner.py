import numpy as np
from typing import List, Tuple

"""
rrt_planner.py
Primary RRT algorithm
"""


class Euclidean:
    """
    Rigid body pose in 2D
    """

    def __init__(self, x: float, y: float, theta: float):
        """
        Constructor for euclidean
        :param x: x position
        :param y: y position
        :param theta: angle for direction of travel
        :return: N/A
        """
        self.x = x
        self.y = y
        self.theta = theta

    def __repr__(self):
        return repr((self.x, self.y, self.theta))

    @classmethod
    def random_pose(cls, x_range: Tuple[float, float],
                    y_range: Tuple[float, float], theta_range: Tuple[float, float]):
        """
        Generates random list of euclidean poses
        :param x_range: [start, end]
        :param y_range: [start, end]
        :param theta_range: [start, end]
        :return: cls(x_new, y_new, theta_new)
        """
        min_value = np.array([x_range[0], y_range[0], theta_range[0]])
        max_value = np.array([x_range[1], y_range[1], theta_range[1]])
        difference = max_value - min_value
        new_value = np.random.rand(3).dot(np.diag(difference)) + min_value

        return cls(x=new_value[0], y=new_value[1], theta=new_value[2])


def local_planner(
        start: Euclidean, target: Euclidean, distance: float, dtheta_max: float) -> Euclidean:
    """
    Finds shortest path to target
    :param start: start Euclidean pose
    :param target: target Euclidean pose
    :param distance: distance to target
    :param dtheta_max: maximum change in direction
    :return: path to target
    """
    p0 = np.array([start.x, start.y])
    p1 = np.array([target.x, target.y])
    dp = p1 - p0
    dp = dp / np.linalg.norm(dp)
    theta = np.arctan2(dp[1], dp[0])
    dtheta = (theta - start.theta + np.pi) % (2 * np.pi) - np.pi
    p = p0 + dp * distance
    if np.abs(dtheta) > dtheta_max:
        return None
    else:
        return Euclidean(p[0], p[1], theta)


class EuclideanTree:
    """
    Tree of euclidean poses.
    """

    def __init__(self, pose: Euclidean):
        self.parent = None
        self.pose = pose
        self.children = []

    def add_child(self, child) -> None:
        assert isinstance(child, EuclideanTree)
        child.parent = self
        self.children.append(child)

    def find_closest(self, pose: Euclidean):
        d = np.linalg.norm(
            [self.pose.x - pose.x, self.pose.y - pose.y, self.pose.theta - pose.theta])
        closest = self

        for child in self.children:
            d_child, closest_to_child = child.find_closest(pose)
            if d_child < d:
                d = d_child
                closest = closest_to_child
        return d, closest

    def path(self) -> List[Euclidean]:
        """
        Recursive function to find the path of the pose to tree root.
        :return: list of poses
        """
        if self.parent is None:
            return [self.pose]
        else:
            return self.parent.path() + [self.pose]


class CircleObstacles:
    """
    Array of circular objects to simulate obstacles.
    """

    def __init__(self, obstacles: np.array):
        assert obstacles.shape[1] == 3
        self.data = obstacles

    @classmethod
    def generate_uniform(cls, x_range: Tuple[float, float],
                         y_range: Tuple[float, float], r_range: Tuple[float, float],
                         samples: int = 10):
        min_value = np.array([x_range[0], y_range[0], r_range[0]])
        max_value = np.array([x_range[1], y_range[1], r_range[1]])
        return_val = np.random.rand(samples, 3).dot(
            np.diag(max_value - min_value)) + min_value
        return cls(return_val)

    def closest_obstacle(self, x: float, y: float) -> Tuple[int, float]:
        """
        Find closest obstacle and distance to coordinates
        :param x: x-coordinate
        :param y: y-coordinate
        :return: min distance to obstacle
        """
        distance = np.linalg.norm(self.data[:, :2] - np.array([
            x, y]), axis=1) - self.data[:, 2]
        i_min = np.argmin(distance)
        return int(i_min), distance[i_min]

    def __repr__(self):
        return repr(self.data)
