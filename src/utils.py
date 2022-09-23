import os
import laspy
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from collections import namedtuple
import pymannkendall as mktest
import matplotlib as mpl

mpl.rcParams['font.size'] = 10
mpl.rcParams['font.sans-serif'] = 'Arial'


# --------------------------------Reading data -----------------------------
def read_las(file_path):
    """Reading point cloud stored with las format
    Args:
        file_path:      str,            path to las file

    Return:
        points:         np.array(n, 3)
    """
    with laspy.open(file_path) as lasfile:
        lasdata = lasfile.read()
        x = lasdata.x
        y = lasdata.y
        z = lasdata.z
        points = np.vstack([x, y, z]).transpose()
    return points


def filter_pts(points, above_ground=1):
    """Filtering ground point clouds
    Args:
        points:             np.array(n, 3), point cloud using x, y, and z
        above_ground:       float,          the width of ground, a buffer parameter

    Return:
        points:             np.array(m, 3), non-ground points
    """
    not_ground = points[:, -1] >= above_ground
    return points[not_ground]


# ---------------------------Resampling strategy------------------------------
def adaptive_sample_pts(points, layer_height=1, return_sample_info=False, is_figure=False,
                        fig_path='./tmp', random_state=666):
    """Resampling point cloud using voxel down sample method, the sample voxel is determined using the points distance and MK-test.
       After the resampling, the point distace along height would have no significant trend.

    Args:
        points:             np.array(n, 3), point cloud including x, y, and z
        layer_height:       float,          the height of layer used to calculate points distance distribution
        return_sample_info: bool,           if ture, return the sample information for sampling
                            process,        including the voxel size, MK-test result before and after sampling
        is_figure:          bool,           whether plot point distance along height before and after resampling
        fig_path:           str,            path to save figure
        random_state:       float,          random state for numpy random function

    Return:
        points:             np.array(m, 3), point cloud that have been downsampled
        sample_info:        namedtuple,     sampling information
    """

    def cal_avg_layer_distance(points_pcd):
        points_diss = np.array(points_pcd.compute_nearest_neighbor_distance())
        points = np.array(points_pcd.points)
        # Resampling point cloud for accelerating the aggeration of the point distance along height
        if len(points) > 50000:
            if random_state is not None:
                np.random.seed(random_state)
            idx = np.random.choice(np.arange(len(points)), 50000, replace=False)
            points = points[idx, :]
            points_diss = points_diss[idx]
        points_z = points[:, -1] - np.min(points[:, -1])
        layers = (points_z / layer_height).astype('int')
        layers_diss = np.vstack([layers, points_diss]).transpose()
        layers_diss = pd.DataFrame(layers_diss, columns=['layers', 'diss'])
        agg = layers_diss.groupby('layers')
        # Calculating the average distane along height
        avg_layer_dis = agg.mean().reset_index()
        avg_layer_dis = avg_layer_dis.sort_values(by='layers')
        return avg_layer_dis

    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(points)
    avg_layer_dis = cal_avg_layer_distance(points_pcd)
    if is_figure:
        if not os.path.exists(os.path.dirname(fig_path)):
            os.makedirs(os.path.dirname(fig_path))
        plt.figure(figsize=(4/2.54, 6/2.54))
        plt.plot(avg_layer_dis['diss'].values, avg_layer_dis['layers'].values)
        plt.tight_layout()
        plt.savefig(fig_path+'_before.jpg', dpi=600)
        plt.clf()
        plt.close()

    # Deriving the maximum point distance after filtering outliers, which is the base for iteratively
    # increasing voxel size
    noise_threshold = avg_layer_dis['diss'].mean() + 2 * avg_layer_dis['diss'].std()
    avg_layer_dis = avg_layer_dis[avg_layer_dis['diss'] < noise_threshold]
    max_dis = avg_layer_dis['diss'].max()

    for ratio in np.arange(2.0, 4.1, 0.1):
        voxel_size = ratio * max_dis
        sampled_points_pcd = points_pcd.voxel_down_sample(voxel_size)
        avg_layer_dis = cal_avg_layer_distance(sampled_points_pcd)
        mk_result = mktest.original_test(avg_layer_dis['diss'].values, alpha=0.05)
        if not mk_result.h:
            break
    if is_figure:
        plt.figure(figsize=(4/2.54, 6/2.54))
        plt.plot(avg_layer_dis['diss'].values, avg_layer_dis['layers'].values)
        plt.tight_layout()
        plt.savefig(fig_path+'_after.jpg', dpi=600)
        plt.clf()
        plt.close()

    if return_sample_info:
        ret = namedtuple('Adaptive_sample_info', ['voxel_size', 'ratio', 'mk_test'])
        return np.array(sampled_points_pcd.points), ret(voxel_size, ratio, mk_result)
    else:
        return np.array(sampled_points_pcd.points)


if __name__ == '__main__':
    pass
