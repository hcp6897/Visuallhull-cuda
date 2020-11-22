from OpenGL.GL import *
from OpenGL.GL import shaders


class ShaderMaterial():
    def __init__(self, vert, frag, uniform):
        self.vertShader = vert
        self.fragShader = frag
        self.uniform = uniform

    def init(self):

        VERTEX_SHADER = shaders.compileShader(
            self.vertShader, GL_VERTEX_SHADER)
        FRAGMENT_SHADER = shaders.compileShader(
            self.fragShader, GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(
            VERTEX_SHADER, FRAGMENT_SHADER)

        self.uniform.init()
        
        return self.shader

    def activate(self):
        # use shader
        shaders.glUseProgram(self.shader)

        self.uniform.update(self.shader)

    def deactivate(self):
        shaders.glUseProgram(0)
