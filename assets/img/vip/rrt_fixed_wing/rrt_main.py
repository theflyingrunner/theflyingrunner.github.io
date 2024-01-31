from rrt_planner import Euclidean, EuclideanTree, CircleObstacles, local_planner
from rrt_plotter import draw_obstacles, draw_pose, draw_path, draw_tree
import matplotlib.pyplot as plt
import numpy as np

"""
rrt_main.py
Executes the RRT code. 
Change input values for obstacle size, number, and range here.
"""


def main():
    obstacles = CircleObstacles.generate_uniform(
        x_range=(1, 9), y_range=(1, 9), r_range=(0.5, 0.8), samples=10)

    start = Euclidean(x=0, y=0, theta=0)
    target = Euclidean(x=10, y=10, theta=1)

    root = EuclideanTree(start)
    node = root
    max_iteration = 100000

    for i in range(max_iteration):
        end_target = np.random.rand() < 0.1
        if end_target:
            temp_target = target
        else:
            temp_target = Euclidean.random_pose((0, 10), (0, 10), (-np.pi, np.pi))
        distance, closest = root.find_closest(temp_target)
        if end_target and distance < 1:
            break
        pose_new = local_planner(
            start=closest.pose, target=temp_target, distance=1, dtheta_max=0.5)
        if pose_new is None:
            continue
        close_i, close_d = obstacles.closest_obstacle(pose_new.x, pose_new.y)
        if close_d < 0:
            continue
        else:
            child = EuclideanTree(pose_new)
            closest.add_child(child)

    rrt_plot = plt.gca()
    h1 = draw_pose(rrt_plot, start, color='crimson', label='start')
    h2 = draw_pose(rrt_plot, target, color='seagreen', label='target')
    h3 = draw_obstacles(rrt_plot, obstacles, color='indigo', label='obstacles')[0]
    h4 = draw_tree(rrt_plot, root, color='k', label='tree')[0]
    h5 = draw_path(rrt_plot, closest.path(), 'y.-', label='path', alpha=0.5, linewidth=5)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(handles=[h1, h2, h3, h4, h5], loc='upper left', ncol=2)
    plt.title('RRT Motion Planner')
    plt.axis([-2, 12, -2, 12])
    rrt_plot.set_aspect('equal', adjustable='box')
    plt.grid()
    plt.savefig('rrt_demo.png')
    plt.close()


main()
