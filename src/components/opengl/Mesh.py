from OpenGL.GL import *

class Mesh():
    def __init__(self, mat, geo):
        self.material = mat
        self.geometry = geo
        self.visible = True
        self.wireframe = False

    def init(self):
        shader = self.material.init()
        self.geometry.shader = shader
        self.geometry.init()

    def draw(self):
        self.material.activate()
        if self.visible:
            if self.wireframe :
                glEnable(GL_CULL_FACE)
                glCullFace(GL_FRONT)
                if self.wireframe is not None :
                    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )

            self.geometry.draw()
        self.material.deactivate()
