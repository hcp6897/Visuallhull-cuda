from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math 

def Spherical2Cartesian(rvh):
    r = rvh[0]
    v = math.radians(rvh[1])
    h = math.radians(rvh[2])
    
    return (r*math.sin(h)*math.cos(v), r*math.sin(h)*math.sin(v), r*math.cos(h))

class Camera:
    def __init__(self, width, height):
        self.setViewport(width, height)

        self.minAzimuthangle = 0
        self.maxAzimuthangle = 180
        self.zoomSensitivity = 0.1

        self.polarAngle = 0
        self.azimuthangle = 1
        self.radius = 3

        self.w = width
        self.h = height

        self.center = (0.0, 0.0, 0.0)
        self.position = (0.0, 0.0, 0.0)

        self.mouseCoord = None

    def setViewport(self, width, height):
        self.w = width
        self.h = height

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        glViewport(0, 0, width, height)
        gluPerspective(60.0, float(width)/float(height), 0.01, 100.0)

    def update(self):

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        glViewport(0, 0, self.w, self.h)
        gluPerspective(60.0, float(self.w)/float(self.h), 0.01, 100.0)

        self.position = Spherical2Cartesian(
            (self.radius, self.polarAngle, self.azimuthangle))
        gluLookAt(self.position[0], self.position[2], self.position[1],  self.center[0],
                  self.center[1], self.center[2],  0.0, 1.0, 0.0)

    def handlGLUTMouseClick(self, button, x, y):
        if(button == GLUT_LEFT_BUTTON):
            if(self.mouseCoord == None):
                self.mouseCoord = (x, y)
            else:
                self.mouseCoord = None

        if(button == 3):
            self.zoom(-1)

        if(button == 4):
            self.zoom(1)

    def handleGLUTMouseMove(self, x, y):
        # print(x, y, self.mouseCoord)
        if(self.mouseCoord != None):

            deltaX = x-self.mouseCoord[0]
            deltaY = self.mouseCoord[1]-y

            self.dragCamera(deltaX, deltaY)

            self.mouseCoord = (x, y)

    def zoom(self, deltaZoom):
        self.radius += self.zoomSensitivity * math.copysign(1, deltaZoom)

    def dragCamera(self, deltaX, deltaY):
        self.polarAngle += deltaX

        if(self.azimuthangle+deltaY > self.minAzimuthangle and self.azimuthangle+deltaY < self.maxAzimuthangle):
            self.azimuthangle += deltaY
