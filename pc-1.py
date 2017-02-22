# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:41:08 2017

@author: Ali.Khalili
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.path as mplPath


work_dir = 'C:/Users/ali.khalili/Desktop/PointCloud/'
data_dir = work_dir + 'xyz_data/'



def read_data(fname, fpath=data_dir, filter_points=None, delim=','):
    '''
    Read data from the csv file and return the dataframe
    fname: filename
    fpath: path to the directory holding the file
    delim: delimeter character
    '''
    # Read pandas dataframe object
    data_df = pd.read_csv(fpath+fname, sep=' ', names=['X','Y','Z'])
    # Filter the data_frame object if necessary
    if filter_points is not None:
        data_df = poly_filter(data_df, filter_points)
    # Return dataframe        
    return data_df



def poly_filter(src_df, points):
    '''
    removes all points in the src_df that are  outside the polygon specified by points 
    it is assumed that each point in points array is forming an edge with the next point,
    and the last point forms an edge with the first point.
    points: np array with shape [numpoints, 2]
    '''
    # Define a polygon object from the points
    polygon = mplPath.Path(points)
    # Create a tmp_dataframe object by copying the src_df
    src_df['Cond'] = polygon.contains_points(src_df.as_matrix(columns=['X','Y']))
    # Return the new data Frame Object
    return src_df[src_df['Cond']==True]
    


def plot_point_cloud(point_df, x_range=None, y_range=None, z_range=None):
    '''
    Plot a point cloud in the pandas dataframe object
    X_range: list indicating the min/max of the plot axis; if None equals the min/max of datapoints
    Y_range: list indicating the min/max of the plot axis; if None equals the min/max of datapoints
    Z_range: list indicating the min/max of the plot axis; if None equals the min/max of datapoints
    '''
    
    # set up figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # copy dataframe object    
    p_df = point_df.copy()
    
    # set axis range for X axis
    if x_range is None:
        ax.set_xlim([p_df['X'].min(), p_df['X'].max()])
    else:
        ax.set_xlim(x_range)
        p_df = p_df[(p_df['X']>=x_range[0]) & (p_df['X']<=x_range[1])]
    # set axis range for Y axis
    if y_range is None:
        ax.set_ylim([p_df['Y'].min(), p_df['Y'].max()])
    else:
        ax.set_ylim(y_range)
        p_df = p_df[(p_df['Y']>=y_range[0]) & (p_df['Y']<=y_range[1])]
    # set axis range for Z axis
    if z_range is None:
        ax.set_zlim([p_df['Z'].min(), p_df['Z'].max()])
    else:
        ax.set_zlim(z_range)
        p_df = p_df[(p_df['Z']>=z_range[0]) & (p_df['Z']<=z_range[1])]
    
    # plot scatter plot with datapoints
    ax.scatter(p_df.X, p_df.Y, p_df.Z, c='b', marker='.', s=2)
    
    # set labels
    ax.set_xlabel('X')    
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
        
    # show the plot
    plt.show()
    
    # return number of points that are plotted
    return len(p_df)



def main():
    pass


if __name__ == '__main__':
    main()