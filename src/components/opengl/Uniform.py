from OpenGL.GL import *
from OpenGL.GL import shaders


class Uniform:
    def __init__(self):
        pass
        self.dict = []
        self.textures = []
        self.floats = []
        self.vec3s = []
        self.mat4s = []

    def init(self):
        for texture in self.textures:
            texture[1].init()

    def addTexture(self, name, texture):
        self.textures.append([name, texture])

    def addFloat(self, name, number):
        self.floats.append([name, number])

    def addvec3(self, name, vec3):
        self.vec3s.append([name, vec3])

    def addMat4(self, name, mat4):
        self.mat4s.append([name, mat4])

    def getValue(self, name):
        for texture in self.textures:
            if(texture[0] == name):
                return texture[1]

        for floatNum in self.floats:
            if(floatNum[0] == name):
                return floatNum[1]

        for vec3 in self.vec3s:
            if(vec3[0] == name):
                return vec3[1]

        for mat4 in self.mat4s:
            if(mat4[0] == name):
                return mat4[1]

    def setValue(self, name, value):
        for texture in self.textures:
            if(texture[0] == name):
                texture[1] = value

        for floatNum in self.floats:
            if(floatNum[0] == name):
                floatNum[1] = value

        for vec3 in self.vec3s:
            if(vec3[0] == name):
                vec3[1] = value

        for mat4 in self.mat4s:
            if(mat4[0] == name):
                mat4[1] = value

    def update(self, shader):
        for texture in self.textures:
            texture[1].activate()
            location = glGetUniformLocation(shader, texture[0])
            glUniform1i(location, texture[1].id)

        for floatNum in self.floats:
            location = glGetUniformLocation(shader, floatNum[0])
            glUniform1f(location, floatNum[1])

        for vec3 in self.vec3s:
            location = glGetUniformLocation(shader, vec3[0])
            glUniform3f(location, vec3[1][0], vec3[1][1], vec3[1][2])

        for mat4 in self.mat4s:
            location = glGetUniformLocation(shader, mat4[0])
            glUniformMatrix4fv(location, 1, GL_TRUE, mat4[1])
