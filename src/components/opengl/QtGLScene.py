from OpenGL.GL import *
from OpenGL.GLU import *

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from .Camera import *


class QtGLScene():
    def __init__(self, openGLWidget):
        self.openGLWidget = openGLWidget

        self.openGLWidget.initializeGL = self.initializeGL
        self.openGLWidget.paintGL = self.paintGL
        self.openGLWidget.resizeGL = self.resizeGL

        self.mouseCoord = None
        self.openGLWidget.mouseMoveEvent = self.mouseMoveEvent
        self.openGLWidget.mousePressEvent = self.mousePressEvent
        self.openGLWidget.mouseReleaseEvent = self.mouseReleaseEvent
        self.openGLWidget.wheelEvent = self.wheelEvent

        format = QSurfaceFormat()
        format.defaultFormat()
        format.setProfile(QSurfaceFormat.CoreProfile)
        self.openGLWidget.setFormat(format)

        self.meshes = []
        self.customRender = []
        size = self.openGLWidget.size()
        self.size = [size.width(), size.height()]

    def initializeGL(self):
        self.openGLWidget.makeCurrent()

        # 設置畫布背景色
        glClearColor(1, 1, 1, 1)
        # 開啓深度測試，實現遮擋關係
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)                                      # 設置深度測試函數
        # GL_SMOOTH(光滑着色)/GL_FLAT(恆定着色)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_BLEND)                                          # 開啓混合
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)           # 設置混合函數
        # 啓用 Alpha 測試
        glEnable(GL_ALPHA_TEST)
        # 設置 Alpha 測試條件爲大於 0.05 則通過
        glAlphaFunc(GL_GREATER, 0.05)
        # 設置逆時針索引爲正面（GL_CCW/GL_CW）
        glFrontFace(GL_CW)
        glEnable(GL_LINE_SMOOTH)                                    # 開啓線段反走樣
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        self.camera = Camera(self.size[0], self.size[1])

        self.openGLWidget.doneCurrent()

    def add(self, mesh):
        self.openGLWidget.makeCurrent()
        mesh.init()
        self.meshes.append(mesh)
        self.openGLWidget.doneCurrent()

    def paintGL(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 設置視口
        self.camera.update()

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        for mesh in self.meshes:
            mesh.draw()

        for customRender in self.customRender:
            customRender()      

    def resizeGL(self, width, height):
        size = self.openGLWidget.size()
        self.size = [size.width(), size.height()]
        self.openGLWidget.makeCurrent()
        self.camera.setViewport(self.size[0], self.size[1])

    def startDraw(self):
        self.openGLWidget.makeCurrent()
        self.openGLWidget.update()

    def endDraw(self):
        self.openGLWidget.doneCurrent()

    def wheelEvent(self, evt):
        self.camera.zoom(evt.angleDelta().y())

    def mouseReleaseEvent(self, evt):
        if(evt.button() == Qt.LeftButton):
            self.mouseCoord = None

    def mousePressEvent(self, evt):
        if(evt.button() == Qt.LeftButton):
            x = evt.localPos().x()
            y = evt.localPos().y()
            self.mouseCoord = (x, y)

    def mouseMoveEvent(self, evt):
        if(self.mouseCoord != None):
            x = evt.localPos().x()
            y = evt.localPos().y()

            deltaX = x-self.mouseCoord[0]
            deltaY = self.mouseCoord[1]-y

            self.camera.dragCamera(deltaX, deltaY)

            self.mouseCoord = (x, y)
