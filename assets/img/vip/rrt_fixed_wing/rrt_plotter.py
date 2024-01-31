import matplotlib.pyplot as plt
import numpy as np
from typing import List
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from rrt_planner import Euclidean, EuclideanTree, CircleObstacles

"""
rrt_plotter.py
Uses rrt_planner.py to draw tree, path, and obstacles.
"""


def draw_obstacles(axes: plt.Axes, obstacles: CircleObstacles, *args, **kwargs):
    col = PatchCollection([plt.Circle((x, y), radius)
                           for x, y, radius in obstacles.data], *args, **kwargs)
    axes.add_collection(col)
    return Patch(*args, **kwargs), col


def draw_pose(axes: plt.Axes, pose: Euclidean, *args, **kwargs):
    return axes.quiver(
        pose.x, pose.y, np.cos(pose.theta), np.sin(pose.theta), *args, **kwargs)


def draw_path(axes: plt.Axes, poses: List[Euclidean], *args, **kwargs):
    return axes.plot([pose.x for pose in poses],
                     [pose.y for pose in poses], *args, **kwargs)[0]


def draw_tree(axes: plt.Axes, node: EuclideanTree, *args, **kwargs):
    branch = None
    for child in node.children:
        branch = axes.plot([node.pose.x, child.pose.x], [node.pose.y, child.pose.y],
                           *args, **kwargs)
        draw_tree(axes, child, *args, **kwargs)

    return branch
