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
import matplotlib.tri as mtri
from sklearn.decomposition import PCA


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
    


def plot_point_cloud(point_df, fname=None, x_range=None, y_range=None, z_range=None, marker_size=0.5):
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
    ax.scatter(p_df.X, p_df.Y, p_df.Z, c='k', marker='.', s=marker_size, facecolors='none', edgecolors='none')
    
    # set labels
    ax.set_xlabel('X')    
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
        
    # save plot to file
    if fname is not None:
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
    form2:  (alpha, theta) where alpha is the angle between the normal vector from origin to the line of 
            intersection of the plane with xy plane and the x axis measured from 0 to 2*pi (0 aligning with x axis), 
            theta is the angle between normal to the plane and the z axis, and delta is the distance from the origin. 
            alpha varies between 0 and 2*pi, theta varies from 0 to pi/2.
    form3:  (norm_strike, dip) where the norm_strike is the angle between the normal vector to the intersection of 
            the plane with xy plane taken fro the origin and the y axis (clockwise is +), and dip is the angle 
            between the normal vector to the plane with xy plane. norm_strike varies between 0 and 2*pi, and dip 
            varies between 0 and pi/2.
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
    # calculate the coefficient determining the point of normal to the line from oirigin
    if (A**2+B**2)!=0:
        t=-D/(A**2+B**2)
    else:
        t=1
    # determine normal vector to the line of intersection of the plane with xy plane taken from the origin
    nx=t*A
    ny=t*B
    # calculate the angle of the normal vector to the x axis
    if A!=0:
        # deermine alpha based on a 360 degree rotation - 0 is the x axis
        alpha = np.math.atan(B/A) # angle of intersection with xy plane with x axis
        # based on the direction of the normal vector determine alpha
        if (nx>=0) and (ny>=0):
            alpha += 0
        if (nx<0) and (ny>=0):
            alpha += np.pi
        if (nx<0) and (ny<0):
            alpha += np.pi
        if (nx>=0) and (ny<0):
            alpha += np.pi*2
    else:
        if ny>=0:
            alpha = np.pi/2
        else:
            alpha = 3*np.pi/2
    theta = np.math.acos(abs(C)/norm_length)
    # Plane equation form 3
    # norm_strike is the angle of the normal to the intersection with xy plane from origin with y axis - clockwise is +
    norm_strike = (5*np.pi/2-alpha)%(2*np.pi) 
    dip = np.pi/2 - theta
    # calculate distance from origin
    delta = abs(D)/norm_length
    # return plane parameters
    return (A, B, C, D), (alpha, theta), (norm_strike,dip), delta
    


def slide_boxes(x_start_stop,y_start_stop,z_start_stop,
                box_size=1.0, overlap=0.5):
    '''
    Takes start/stop coordinates, box_size and overlap and returns list of box coordinates.
    x_start_stop: start and end position in x direction (array of size 2)
    y_start_stop: start and end position in y direction (array of size 2)
    z_start_stop: start and end position in z direction (array of size 2)
    xyz_box: size of the box in x, y and y directions. 
    xy_overlap: amount of overlap between windows in x, y and z directions
    '''
    
    # Compute set box_size and overlap in all directions - 3-tuples
    xyz_box=(box_size, box_size, box_size)
    xyz_overlap=(overlap, overlap, overlap)
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1]-x_start_stop[0]
    yspan = y_start_stop[1]-y_start_stop[0]
    zspan = z_start_stop[1]-z_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_per_step = xyz_box[0]*(1-xyz_overlap[0])
    ny_per_step = xyz_box[1]*(1-xyz_overlap[1])
    nz_per_step = xyz_box[2]*(1-xyz_overlap[2])
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_per_step) - 1
    ny_windows = np.int(yspan/ny_per_step) - 1
    nz_windows = np.int(zspan/nz_per_step) - 1
    # Initialize a list to append window positions to
    box_list = []
    # Loop through finding x, y and z window positions
    for zs in range(nz_windows):
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_per_step + x_start_stop[0]
                endx = startx + xyz_box[0]
                starty = ys*ny_per_step + y_start_stop[0]
                endy = starty + xyz_box[1]
                startz = zs*nz_per_step + z_start_stop[0]
                endz = startz + xyz_box[2]
                # Append window position to list
                box_list.append(((startx, starty, startz), (endx, endy, endz)))
    # Return the list of windows
    return box_list



def build_plane_list(points_df, plane_list, point_list, triangles=None, dist_cutoff=0.5):
    '''
    For every 2 points in the points_df calculate the plunge and azimuth and delta and form a 
    3D histogram.
    points_df: dataframe containing x, y, z of all points in the point cloud
    plane_list: array/list containing parameters for all of the planes that are extracted from current point cloud
    triangles: the list of triangles to explore
    dist_cutoff: distance cut_off for pruning triangles
    '''
    
    for [irow,jrow,krow] in triangles:
        i = points_df.index[irow]
        j = points_df.index[jrow]
        k = points_df.index[krow]
        # Only points if their distances along each major axis is less than cut-off
        tmp_df = points_df[(points_df.index==i) | (points_df.index==j) | (points_df.index==k)]
        if tmp_df.X.ptp()<=dist_cutoff:          
            if tmp_df.Y.ptp()<=dist_cutoff:
                if tmp_df.Z.ptp()<=dist_cutoff:
                    # calculate line parameters
                    p0 = (tmp_df['X'][i], tmp_df['Y'][i], tmp_df['Z'][i])
                    p1 = (tmp_df['X'][j], tmp_df['Y'][j], tmp_df['Z'][j])
                    p2 = (tmp_df['X'][k], tmp_df['Y'][k], tmp_df['Z'][k])
                    (A, B, C, D), (alpha, theta), (norm_strike,dip), delta = plane_params(p0, p1, p2)
                    # append results to the list/array
                    plane_list.append([A, B, C, D, alpha, theta, norm_strike, dip, delta, i, j, k])
                    point_list.append([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]])
        
    return plane_list, point_list
    
    
    
def read_data_and_process(data_frame=None, verbose=True):
    '''
    Read points cloud data within a specified zone and extract all plane parameters
    data_frame: if provided, the data in the dataframe will be used without filtering instead of reading from file.
    verbose: print additional details if True
    '''
    # Define boundary points and read/extract point cloud
    bounary_points = [(543214,5698566),(543233,5698568),(543233,5698581),(543214,5698579)]
    # Reading and initial filtering of the point cloud
    if data_frame is None:
        if verbose:
            print('Reading data....')
        t_start = time.time()
        df1, translation = read_data('hwy1w.pts', delim=' ', filter_points=bounary_points)    
        t_end = time.time()
        if verbose:
            print('Reading Complete... - Time to Read: {:3f} seconds'.format(t_end-t_start))
    else:
        df1 = data_frame
    # Calculate box list
    box_list = slide_boxes([df1['X'].min(), df1['X'].max()], 
                           [df1['Y'].min(), df1['Y'].max()], 
                           [df1['Z'].min(), df1['Z'].max()],
                           box_size=1.0, overlap=0.2)
    # Initialize the objects to keep track of planes, points on planes and number of points in each sliding box
    plane_list = []
    point_list = []
    points_count = []
    # Iterate through box_list and filter only points that are inside the box and process them
    for box in tqdm(box_list):
        tmp_df = df1[(df1['X']>=box[0][0]) & (df1['X']<=box[1][0]) & \
                     (df1['Y']>=box[0][1]) & (df1['Y']<=box[1][1]) & \
                     (df1['Z']>=box[0][2]) & (df1['Z']<=box[1][2])]
        # process the reduced dataframe if a minimum number of points exist
        if len(tmp_df)>3:
            # Keep track of the number of datapoints in each sliding box
            points_count.append(len(tmp_df))
            # create ndarray and perform PCA analysis on datapoints
            data_vals = tmp_df.values 
            pca = PCA()
            pca.fit_transform(data_vals)
            # Triangulate parameter space to determine the triangles
            tri = mtri.Triangulation(data_vals[:,0], data_vals[:,1])            
            # Add to the 3d histogram
            plane_list, point_list = build_plane_list(tmp_df, plane_list, point_list, tri.triangles, dist_cutoff=0.5)
        
    return df1, plane_list, point_list, points_count
    


def test(strike_norm, dip):
    
    #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #
    ax.hist2d(strike_norm, dip, bins=100)
    plt.colorbar()
    plt.show()


def main():
    pass


if __name__ == '__main__':
    main()