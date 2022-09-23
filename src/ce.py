import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from utils import read_las, filter_pts, adaptive_sample_pts


def cal_entropy(pts, is_resample=True, bandwidth=0.2, grid_size=0.1, is_fig=False, fig_path='./tmp', save_data=False, return_bw=False):
    """ Calculate the continuous entropy by using kernel density estimation and project 3D point cloud
        into plane include (x-y, x-z, y-z)

    Args:
        pts:            np.array(n, 3),  point cloud
        is_resample:    bool,            wheter using MK-test resampling strategy
        band_width:     float,           band width of kernel density, if band_width is None, it would be calculated using grid search
        grid_size:      float,           grid size for calcualting the integral of continous entropy
        is_fig:         bool,            whether output figure
        fig_path:       str,             path to output figure
        save_data:      bool,            whether save the density data
        return_bw:      bool,            whether return band width

    Return:
        c_entropys:                 pd.DataFrame    entropy include three planes
        band_width(optional):       list, band width for generating density
    """
    if len(pts.shape) == 1:
        pts = pts.reshape(-1, 1)
    if is_fig:
        plt.figure()
        if not os.path.exists(os.path.dirname(fig_path)):
            os.makedirs(os.path.dirname(fig_path))

    if is_resample:
        pts = adaptive_sample_pts(pts, is_figure=True, fig_path=fig_path)

    pts_min = np.min(pts, axis=0)
    pts_max = np.max(pts, axis=0)
    c_entropys = []
    bw = bandwidth
    for axis_0, axis_1 in zip([0, 0, 1], [1, 2, 2]):
        print('{}{}'.format('xyz'[axis_0], 'xyz'[axis_1]))
        kde = KernelDensity(kernel='gaussian', bandwidth=bw, rtol=0.001).fit(pts[:, [axis_0, axis_1]])
        if grid_size is None:
            grid_size = bw
        col_grids = int((pts_max[axis_0] - pts_min[axis_0] + 8*bw) / grid_size) + 2     # Plusing 8bw for generating appropriate border of kernal density estimation
        col_locs = pts_min[axis_0] - 4*bw + (np.arange(col_grids)-0.5) * grid_size

        row_grids = int((pts_max[axis_1] - pts_min[axis_1] + 8*bw) / grid_size) + 2
        row_locs = pts_max[axis_1] + 4*bw - (np.arange(row_grids)-0.5) * grid_size
        X, Y = np.meshgrid(col_locs, row_locs)
        xy = np.vstack([X.ravel(), Y.ravel()]).T

        density = np.exp(kde.score_samples(xy))     # The density of KernelDensity in sklearn is logirithm
        density_nozero = density[density > 0]       # Removing the density whose value is zero
        c_entropy = -1 * np.sum(density_nozero*np.log(density_nozero)*grid_size*grid_size)

        if is_fig:
            density_matrix = density.reshape(row_grids, col_grids)
            plt.imshow(density_matrix)
            plt.tight_layout()
            plt.savefig(fig_path + '_{}{}.jpg'.format('xyz'[axis_0], 'xyz'[axis_1]))
            plt.clf()
            if save_data:
                np.save(fig_path + '_{}{}.npy'.format('xyz'[axis_0], 'xyz'[axis_1]), density_matrix)

        c_entropys.append(c_entropy)
    c_entropys = np.array(c_entropys).reshape(1, -1)
    c_entropys = pd.DataFrame(c_entropys, columns=['ce_xy', 'ce_xz', 'ce_yz'])
    c_entropys['ce'] = np.sqrt(np.sum(c_entropys[['ce_xy', 'ce_xz', 'ce_yz']].values ** 2))

    if is_fig:
        plt.close()
    return c_entropys


if __name__ == '__main__':
    file_path = '../data/test.las'         # Tha path to point cloud file
    pts = read_las(file_path)

    # Removing the ground points. Note that the point cloud must be normalized before this step
    pts = filter_pts(pts)

    # Calculating canopy entropy
    c_entropys = cal_entropy(pts, is_fig=True, fig_path=os.path.join('../data', 'figure/test'))
    # Saving calculated canopy entropy to file
    c_entropys.to_csv('../data/test.csv', index=False)

    # Calculating canopy entropy
    c_entropys = cal_entropy(pts, is_resample=False, is_fig=True, fig_path=os.path.join('../data', 'figure/test_nosample'))
    # Saving calculated canopy entropy to file
    c_entropys.to_csv('../data/test_nosample.csv', index=False)