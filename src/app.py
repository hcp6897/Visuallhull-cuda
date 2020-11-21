from numba import cuda
import numpy as np
import math,json,cv2,os
from time import time
from skimage import measure
import open3d as o3d

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
                if imgs[i][int(u*w)][int(h*v)][0] > 10 or imgs[i][int(u*w)][int(h*v)][1] > 10 or imgs[i][int(u*w)][int(h*v)][2] > 10 :
                    count += 1

        if count == len(cameras):
            tsdf[idx]= 0
        else:
            tsdf[idx] = 1


def visuallhull():

    camParams=[]
    silhouetteImgs=[]  

    # 讀取圖片跟cameraPose
    folder = '../resources'
    with open(folder+'./camera.json') as f:
        data = json.load(f)
    for cam in data['arr']:
        #print(cam)
        camParams.append([
            [cam['mat']['e00'],cam['mat']['e01'],cam['mat']['e02'],cam['mat']['e03']],
            [cam['mat']['e10'],cam['mat']['e11'],cam['mat']['e12'],cam['mat']['e13']],
            [cam['mat']['e20'],cam['mat']['e21'],cam['mat']['e22'],cam['mat']['e23']],
            [cam['mat']['e30'],cam['mat']['e31'],cam['mat']['e32'],cam['mat']['e33']],
        ])
        silhouetteImgs.append(
            cv2.imread(folder+cam['img'])
        )
    
    n = 512
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
    print("gpu vector add time " + str(time() - start))
    
    start = time()
    result = gpu_result.copy_to_host()

    verts, faces, normals, values = measure.marching_cubes(result.reshape((n,n,n)),0)
    print("marching cube time" + str(time() - start))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh('./mesh2.ply', mesh)

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

    #     npPcd.append([posx,posy,posz])
    #     if value == 1:
    #         npColor.append([1.0,0.0,0.0])
    #     else:
    #         npColor.append([0.0,1.0,0.0])

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.array(npPcd))
    # pcd.colors = o3d.utility.Vector3dVector(np.array(npColor))
    # o3d.io.write_point_cloud('./pcd.ply', pcd)


if __name__ == "__main__":
    visuallhull()
