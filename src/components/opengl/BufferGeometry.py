from OpenGL.arrays import vbo
from OpenGL.GL import *
import numpy as np
import time
import math 
import sys
import bisect 
import numpy as np
import open3d as o3d

def DEBUG_SHOWATTRIBUTE(obj):
    for att in dir(obj):
        print (att, getattr(obj,att))

class BufferGeometry:
    def __init__(self):       

        self.mesh=None
        
    def isNotVaild(self):
        return self.mesh is None or self.vao is None

    def readObj(self,filename):

        # read uv
        self.mesh = o3d.io.read_triangle_mesh(filename)
        self.mesh.compute_vertex_normals()

        self.init()
        return True

    def getNormalizeMat(self):

        vertices = np.asarray(self.mesh.vertices, dtype='f')
        # 找points boundingbox對角線長度,做scale
        upperbound = np.amax(vertices, 0)
        lowerbound = np.amin(vertices, 0)
        dist = np.linalg.norm(upperbound - lowerbound)
        scale = 1/dist

        # 找找points重心,做offset
        length = vertices.shape[0]
        center_x = np.sum(vertices[:, 0])/length
        center_y = np.sum(vertices[:, 1])/length
        center_z = np.sum(vertices[:, 2])/length

        return [
            [-scale, 0.0, 0.0, center_x*scale],
            [0.0, -scale, 0.0, center_y*scale],
            [0.0, 0.0, scale, -center_z*scale],
            [0.0, 0.0, 0.0, 1.0]
        ]       

    def genMeshVAO(self):

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        origion_vertices = np.asarray(self.mesh.vertices, dtype='f')
        origion_vertices_normals = np.asarray(self.mesh.vertex_normals, dtype='f')

        verticesArr = []
        vertices_normalsArr = []

        self.indices = np.asarray(self.mesh.triangles, dtype='i')
        for index in self.indices:
            for i in index:
                verticesArr.append(origion_vertices[i])
                vertices_normalsArr.append(origion_vertices_normals[i])

        vertices = np.array(verticesArr)
        vertices_normals = np.array(vertices_normalsArr)
        vertices_uvs = np.asarray(self.mesh.triangle_uvs, dtype='f')

        vertexPositions = vbo.VBO(vertices)
        verticesNormals = vbo.VBO(vertices_normals)
        verticesUvs = vbo.VBO(vertices_uvs)
        
        vertexPositions.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointerf(vertexPositions)
        vertexPositions.unbind()

        verticesNormals.bind()
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointerf(verticesNormals)
        verticesNormals.unbind()

        verticesUvs.bind()
        glClientActiveTexture(GL_TEXTURE0)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glTexCoordPointerf(verticesUvs)
        verticesUvs.unbind()

        glBindVertexArray(0)
            
    def init(self):
        if self.mesh is not None:        
            self.genMeshVAO()

    def draw(self):
        if self.isNotVaild():
            return

        glBindVertexArray(self.vao)    
        glDrawArrays(GL_TRIANGLES, 0 ,len(self.indices)*3)
        glBindVertexArray(0)