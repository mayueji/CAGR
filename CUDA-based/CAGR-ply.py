from time import time
import argparse
import os
import numpy as np
import trimesh
import os

def ply_to_obj(input_file, output_file):
    mesh = trimesh.load_mesh(input_file)
    mesh.export(output_file)

def find_nearest_points_and_compute_inner_product(points1, normals1, points2, normals2):
    kdtree = KDTree(points2)
    _, indices = kdtree.query(points1)
    inner_products = np.sum(normals1 * normals2[indices], axis=1)

    return inner_products
def compute_shortest_distances(point_cloud_A, point_cloud_B):
    kdtree_B = cKDTree(point_cloud_B) 

    distances, _ = kdtree_B.query(point_cloud_A)  
    return distances

def isexist(name, path=None):
    if path is None:
        path = os.getcwd()
    if os.path.exists(path + '/' + name):
        print("Under the path: " + path + '\n' + name + " is exist")
        return True
    else:
        if (os.path.exists(path)):
            print("Under the path: " + path + '\n' + name + " is not exist")
        else:
            print("This path could not be found: " + path + '\n')
        return False
N=5000
currentpath = os.getcwd().replace("\\",'/') 

result= os.path.join(currentpath,'input').replace("\\",'/')
isExists = os.path.exists(result) 

if not isExists: 
    os.mkdir(result) 
file_name=r'./data/bunny.ply'
file_name2=os.path.basename(file_name)
mesh = trimesh.load_mesh(file_name)
points,faces=trimesh.sample.sample_surface_even(mesh,N)
mu, sigma = 0, 1
s = np.random.normal(mu, sigma, points.shape) *0.000
normal=mesh.face_normals[faces]
PointCloud=np.array(points)+s
np.savetxt(result+'/'+file_name2[:-4]+'.xyz',  np.c_[PointCloud],fmt='%f',delimiter=' ')
sample=result+'/'+file_name2[:-4]+'.xyz'
A=r"python main_CAGR.py"
recon_cmdCAGR= f"{A} {sample}"+" --width_config l0 --tqdm "
os.system(recon_cmdCAGR)