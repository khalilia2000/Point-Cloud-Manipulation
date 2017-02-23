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
import itertools
from scipy import linspace
from scipy.misc import comb
import time
from tqdm import tqdm


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
    return data_df-(data_df.min()+data_df.max())/2, (data_df.min()+data_df.max())/2



def poly_filter(src_df, points):
    '''
    Remove all points in the src_df that are outside the polygon specified by points. It is assumed 
    that each point in points array is forming an edge with the next point, and the last point forms 
    an edge with the first point. 
    points: np array with shape [numpoints, 2]
    '''
    # Define a polygon object from the points
    polygon = mplPath.Path(points)
    # Create a tmp_dataframe object by copying the src_df
    src_df['Cond'] = polygon.contains_points(src_df.as_matrix(columns=['X','Y']))
    # Return the new data Frame Object
    return (src_df[src_df['Cond']==True])[['X','Y','Z']]
    


def plot_point_cloud(point_df, fname, x_range=None, y_range=None, z_range=None):
    '''
    Plot a point cloud given data points in a pandas dataframe object
    X_range: list indicating the min/max of the plot axis; if None equals the min/max of datapoints
    Y_range: list indicating the min/max of the plot axis; if None equals the min/max of datapoints
    Z_range: list indicating the min/max of the plot axis; if None equals the min/max of datapoints
    fname: filename to use to save the image to
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
    ax.scatter(p_df.X, p_df.Y, p_df.Z, c='k', marker='.', s=0.5, facecolors='none', edgecolors='none')
    
    # set labels
    ax.set_xlabel('X')    
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
        
    # save plot to file
    fig.savefig(work_dir+fname, dpi=600)
    # show the plot
    plt.show()
    
    
    # return number of points that are plotted
    return len(p_df)



def line_params(p0, p1):
    '''
    Provides the equation of the line in all forms specified below, given 2 points p0 and, p1:
    form1:  (l,m,n) where (X-x0)/l=(Y-y0)/m=(Z-z0)/n=t, 
    form2:  (plunge, azimuth) where plunge (or dip) is the angle with xy plane (between 0 and pi/2 radians), 
            and azimuth is the angle of the projection on xy plane with the y axis (betweeb 0 to 2*pi radians).
    p0 and p1 are 3-tuples with x, y, z coordinates
    '''
    # Line equation form 1
    l = (p0[0]-p1[0])
    m = (p0[1]-p1[1])        
    n = (p0[2]-p1[2])    
    # Line equation form 2
    norm_length_3D = ((l**2+m**2+n**2)**0.5)
    plunge = np.pi/2-np.math.acos(abs(n)/norm_length_3D)
    if l!=0:
        azimuth = np.pi/2-np.math.atan(m/l)
    else:
        azimuth = 0
    norm_length_2D = ((l**2+m**2)**0.5)
    delta = abs(p1[0]*p0[1]-p1[1]*p0[0])/(norm_length_2D)
    
    # return results
    return (l,m,n), (plunge,azimuth,delta)
    


def plane_params(p0, p1, p2):
    '''
    Provide the equation of the plane in all forms specified below, given 3 points p0, p1 and p2:
    form1:  (A, B, C, D) where Ax+By+Cz+D=0
    form2:  (theta, alpha) where theta is the angle between normal to the plane and the z axis,
            alpha is the angle between the intersection of the plane with xy plane and the x axis, and
            delta is the distance from the origin. theta varies from 0 to 90 degrees, and alpha varies 
            from -90 to 90 degrees.
    form3:  (strike, dip) where the strike is the angle between the intersection of the plane with xy plane 
            and the y axis, and dip is the angle between the normal vector to the plane with xy plane. strike
            varies between -90 and 90 degrees, and dip varies between 0 and 90 degrees
    return delta - the distance from origin - to be used along with forms 2 and 3 above.
    p0, p1 and p2 are 3-tuples with x, y, z coordinates
    '''
    # Plane equation form 1
    A = (p1[1]-p0[1])*(p2[2]-p0[2])-(p2[1]-p0[1])*(p1[2]-p0[2])
    B = (p2[0]-p0[0])*(p1[2]-p0[2])-(p1[0]-p0[0])*(p2[2]-p0[2])
    C = (p1[0]-p0[0])*(p2[1]-p0[1])-(p2[0]-p0[0])*(p1[1]-p0[1])
    D = -p0[0]*A-p0[1]*B-p0[2]*C
    norm_length = ((A**2+B**2+C**2)**0.5)
    # Plane equation form 2    
    if B!=0:
        alpha = np.math.atan(A/B) # angle of intersection with xy plane with x axis
    else:
        alpha = np.pi/2
    theta = np.math.acos(abs(C)/norm_length)
    # Plane equation form 3
    strike = np.pi/2-alpha # angle of intersection with xy plane with y axis    
    dip = np.pi/2 - theta
    # calculate distance from origin
    delta = abs(D)/norm_length
    # return plane parameters
    return (A, B, C, D), (alpha, theta), (strike,dip), delta
    


def build_3D_hist(points_df, 
                  alpha_bins=(-np.pi/2,np.pi/2,180),
                  theta_bins=(0,np.pi/2,90), 
                  delta_bins=(0,30,300), 
                  dist_cutoff=0.5):
    '''
    For every 2 points in the points_df calculate the plunge and azimuth and delta and form a 
    3D histogram.
    points_df: dataframe containing x, y, z of all points
    plunge_bins: 3-tuple containing (min, max, nbins) for plunge
    azimuth_bins: 3-tuple containing (min, max, nbins) for azimuth
    delta_bins: 3-tuple containing (min, max, nbins) for delta
    '''
    # Sort points_df based on X, Y, and Z
    points_df.sort_values(['X', 'Y', 'Z'], ascending=[1,1,1], inplace=True)    
    
    # Calculate the borders and midpoints of each bin
    theta_borders = linspace(theta_bins[0], theta_bins[1], theta_bins[2]+1)
    theta_mids = (theta_borders[:-1]+theta_borders[1:])/2
    alpha_borders = linspace(alpha_bins[0], alpha_bins[1], alpha_bins[2]+1)
    alpha_mids = (alpha_borders[:-1]+alpha_borders[1:])/2
    delta_borders = linspace(delta_bins[0], delta_bins[1], delta_bins[2]+1)
    delta_mids = (delta_borders[:-1]+delta_borders[1:])/2
    
    # Initialize the 3D histogram
    hist_3D = np.zeros((alpha_bins[2], theta_bins[2], delta_bins[2]))
    
    # keep track of time
    t_start = time.time()    
    # Iterate through all 2 point combinations
    num_points = 3
    for (i,j,k) in tqdm(itertools.combinations(points_df.index, num_points), total=int(comb(len(points_df), num_points))):
        # Only points if their distances along each major axis is less than cut-off
        if True:
            tmp_df = points_df[(points_df.index==i) | (points_df.index==j) | (points_df.index==k)]
            if tmp_df.X.ptp()<=dist_cutoff:          
                if tmp_df.Y.ptp()<=dist_cutoff:
                    if tmp_df.Z.ptp()<=dist_cutoff:
                        # calculate line parameters
                        p0 = (tmp_df['X'][i], tmp_df['Y'][i], tmp_df['Z'][i])
                        p1 = (tmp_df['X'][j], tmp_df['Y'][j], tmp_df['Z'][j])
                        p2 = (tmp_df['X'][k], tmp_df['Y'][k], tmp_df['Z'][k])
                        (A, B, C, D), (alpha, theta), (strike,dip), delta = plane_params(p0, p1, p2)
                        # calculate the corresponding index for the hist_3D object
                        a_loc = int(np.floor((alpha-alpha_bins[0])/(alpha_bins[1]-alpha_bins[0])*alpha_bins[2]))
                        t_loc = int(np.floor((theta-theta_bins[0])/(theta_bins[1]-theta_bins[0])*theta_bins[2]))
                        d_loc = int(np.floor((delta-delta_bins[0])/(delta_bins[1]-delta_bins[0])*delta_bins[2]))
                        # increase the bin value
                        hist_3D[a_loc, t_loc, d_loc] += 1
    # keep track of time
    t_end = time.time()

    print('total time = {:.3f} seconds'.format(t_end-t_start))
    
    return hist_3D, theta_mids, alpha_mids, delta_mids
    
    


def main():
    pass


if __name__ == '__main__':
    main()