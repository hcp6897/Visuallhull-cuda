from numba import cuda
import numpy as np
import math,json,cv2,os
from time import time
from skimage import measure
import open3d as o3d

from args import build_argparser
args = build_argparser().parse_args()

@cuda.jit
def checkVoxel(cameras, imgs, tsdf, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n**3:

        z = (idx/(n*n))
        zRemain = idx%(n*n)
        y = n - zRemain/n
        x = zRemain%n
        
        posx = (x/n-0.5)*5
        posy = (y/n-0.5)*5
        posz = (z/n-0.5)*5
        posw = 1.0

        count = 0
        #for i in range(len(cameras)):
        for i in range(len(cameras)):        
            projectW = cameras[i][3][0]*posx+cameras[i][3][1]*posy+cameras[i][3][2]*posz+cameras[i][3][3]*posw
            projectX = (cameras[i][0][0]*posx+cameras[i][0][1]*posy+cameras[i][0][2]*posz+cameras[i][0][3]*posw)/projectW
            projectY = (cameras[i][1][0]*posx+cameras[i][1][1]*posy+cameras[i][1][2]*posz+cameras[i][1][3]*posw)/projectW
            projectZ = (cameras[i][2][0]*posx+cameras[i][2][1]*posy+cameras[i][2][2]*posz+cameras[i][2][3]*posw)/projectW        
            u,v = projectY*0.5+0.5,projectX*0.5+0.5
            w = len(imgs[i])
            h = len(imgs[i][0])

            if u<=1.0 and u>=0.0 and v<=1.0 and v>=0.0:
                u = 1-u
                if imgs[i][int(u*w)][int(h*v)][0] > 10 or imgs[i][int(u*w)][int(h*v)][1] > 10 or imgs[i][int(u*w)][int(h*v)][2] > 10 :
                    count += 1

        if count == len(cameras):
            tsdf[idx]= 1
        else:
            tsdf[idx] = 0


def visuallhull(idx,cams):

    camParams=[]
    silhouetteImgs=[]
    
    # 讀取圖片跟cameraPose
    folder = os.path.dirname(args.config)
    for cam in cams:
        #print(cam)
        camParams.append([
            [cam['world2screenMat']['e00'],cam['world2screenMat']['e01'],cam['world2screenMat']['e02'],cam['world2screenMat']['e03']],
            [cam['world2screenMat']['e10'],cam['world2screenMat']['e11'],cam['world2screenMat']['e12'],cam['world2screenMat']['e13']],
            [cam['world2screenMat']['e20'],cam['world2screenMat']['e21'],cam['world2screenMat']['e22'],cam['world2screenMat']['e23']],
            [cam['world2screenMat']['e30'],cam['world2screenMat']['e31'],cam['world2screenMat']['e32'],cam['world2screenMat']['e33']],
        ])
        silhouetteImgs.append(
            cv2.imread(os.path.join(folder,cam['img'][idx]))
        )
    
    n = args.resolution
    print('resolution: {0}*{0}*{0}'.format(n))
    x = np.array(camParams)
    y = np.array(silhouetteImgs)

    # 拷貝數據到設備端
    camerasMats = cuda.to_device(x)
    imgs = cuda.to_device(y)

    # 在顯卡設備上初始化一塊用於存放GPU計算結果的空間
    gpu_result = cuda.device_array(n**3)
    threads_per_block = 1024
    blocks_per_grid = math.ceil(n**3 / threads_per_block)

    start = time()
    checkVoxel[blocks_per_grid, threads_per_block](camerasMats, imgs, gpu_result, n)
    cuda.synchronize()
    print("gpu 1st checkvoxel time " + str(time() - start))
    
    start = time()
    checkVoxel[blocks_per_grid, threads_per_block](camerasMats, imgs, gpu_result, n)
    cuda.synchronize()
    print("gpu 2nd checkvoxel time " + str(time() - start))

    start = time()
    checkVoxel[blocks_per_grid, threads_per_block](camerasMats, imgs, gpu_result, n)
    cuda.synchronize()
    print("gpu 3rd checkvoxel time " + str(time() - start))

    
    start = time()
    result = gpu_result.copy_to_host()

    verts, faces, normals, values = measure.marching_cubes(result.reshape((n,n,n)),0)
    verts/=n
    verts-=0.5
    verts*=5
    x = np.array(verts[:,0])
    verts[:,0] = verts[:,2]
    verts[:,2] = x
    print("marching cube time " + str(time() - start))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(os.path.join(args.output,'./visaulhullMesh_{0}_{1}.ply'.format(idx,n)), mesh)

    # # Debug view TSDF
    # npPcd = []
    # npColor = []
    # for idx,value in enumerate(result):
    #     z = (idx/(n*n))
    #     zRemain = idx%(n*n)
    #     y = n - zRemain/n
    #     x = zRemain%n
        
    #     posx = (x/n-0.5)*3
    #     posy = (y/n-0.5)*3
    #     posz = (z/n-0.5)*3

    #     if not value == 1:
    #         npColor.append([0.0,1.0,0.0])
    #         npPcd.append([posx,posy,posz])


    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.array(npPcd))
    # pcd.colors = o3d.utility.Vector3dVector(np.array(npColor))
    # o3d.io.write_point_cloud('./pcd.ply', pcd)


if __name__ == "__main__":
    with open(args.config) as f:
        data = json.load(f)
        
    visuallhull(0,data['camera'])
    visuallhull(1,data['camera'])
    visuallhull(2,data['camera'])
    visuallhull(3,data['camera'])

