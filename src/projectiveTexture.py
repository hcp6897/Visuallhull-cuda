import cv2
import sys
import json
from dotmap import DotMap

from components.qt.window import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog

from components.opengl.QtGLScene import *
from components.opengl.ShaderMaterial import *
from components.opengl.Texture import *
from components.opengl.Mesh import *
from components.opengl.Uniform import *
from components.opengl.FrameBuffer import *

import components.opengl.shaderLib.ProjectiveTextures as ProjectiveTextures
import components.opengl.shaderLib.DepthMap as DepthMap
from components.opengl.BufferGeometry import *
class PojectiveTextureMesh():
    def __init__(self,scene,count=3):
        
        self.scene = scene

        self.geoModel = BufferGeometry()

        self.textureCount = count
        # prepare depth test textures when projecting RGB textures
        self.depthMaps = []
        self.depthMapModels = []

        self.initProjectDepthTest()
        self.initMesh()       

    def initMesh(self):
        self.uniformModel = Uniform()        
        for i in range(self.textureCount):
            self.uniformModel.addFloat('wstep',1e-2)
            self.uniformModel.addFloat('hstep',1e-2)
            self.uniformModel.addTexture('depthMap[{0}]'.format(i),self.depthMaps[i])
            self.uniformModel.addTexture('projectTex[{0}]'.format(i),Texture(np.zeros((2,2))))
            self.uniformModel.addvec3('cameraPose[{0}]'.format(i),np.array([0,0,0]))
            self.uniformModel.addMat4('projectMat[{0}]'.format(i),np.identity(4))

        self.uniformModel.addMat4('normalizeMat', np.identity(4))
        matModel = ShaderMaterial(ProjectiveTextures.vertex_shader,
                            ProjectiveTextures.fragment_shader(self.textureCount),
                            self.uniformModel)

        model = Mesh(matModel, self.geoModel)
        model.wireframe = False
        self.scene.add(model)

    def initProjectDepthTest(self):                    
        self.scene.startDraw()

        for i in range(self.textureCount):
            self.depthMaps.append(FrameBuffer(self.scene.size))

            depthMapUniform = Uniform()
            depthMapUniform.addMat4('worldToSrcreen',np.identity(4))
            matdepthMapModel = ShaderMaterial(DepthMap.vertex_shader,DepthMap.fragment_shader,depthMapUniform)
            depthMapModel = Mesh(matdepthMapModel, self.geoModel)
            depthMapModel.init()
            self.depthMapModels.append(depthMapModel)

        self.scene.endDraw()

    def renderDepthTestTextures(self):
        for i in range(self.textureCount):
            self.depthMaps[i].updateResolution(self.scene.size)
            self.depthMaps[i].startDraw()
            self.depthMapModels[i].draw()
            self.depthMaps[i].endDraw()

    def loadPly(self,filename):
        self.scene.startDraw()
        sucess = self.geoModel.readObj(filename)
        if sucess:
            self.uniformModel.setValue('normalizeMat',self.geoModel.getNormalizeMat())
        self.scene.endDraw()

    def loadTexturesAndMats(self,camParams,silhouetteImgs,camPoses):
        self.scene.startDraw()

        if len(silhouetteImgs)>0 :
            self.uniformModel.setValue('wstep',1/silhouetteImgs[0].shape[0])
            self.uniformModel.setValue('hstep',1/silhouetteImgs[0].shape[0])

        for i in range(self.textureCount):
            silhouetteImg = Texture(silhouetteImgs[i])
            silhouetteImg.init()
            
            self.uniformModel.setValue('projectTex[{0}]'.format(i),silhouetteImg)
            self.uniformModel.setValue('projectMat[{0}]'.format(i),camParams[i])
            self.uniformModel.setValue('cameraPose[{0}]'.format(i),camPoses[i])
            self.depthMapModels[i].material.uniform.setValue('worldToSrcreen',camParams[i])

        self.scene.endDraw()

    def render(self):
        self.uniformModel.setValue('viewPos', [
        self.scene.camera.position[0],
        self.scene.camera.position[1],
        self.scene.camera.position[2]]
        )

from args import build_argparser
args = build_argparser().parse_args()

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
scene = QtGLScene(ui.openGLWidget)
MainWindow.show()

# projective texture model
mesh = PojectiveTextureMesh(scene)
scene.customRender.append(mesh.renderDepthTestTextures)

def mainloop():
    mesh.render()
    scene.startDraw()
    scene.endDraw()

timer = QTimer(MainWindow)
timer.timeout.connect(mainloop)
timer.start(1)

def importResource():
    qfd = QFileDialog()
    filter = "(*.json *.ply)"
    filenames,_ = QFileDialog.getOpenFileNames(qfd,'import', './', filter)

    camParams=[]
    camPoses=[]
    silhouetteImgs=[]

    for filename in filenames:

        if '.json' in filename:
            folder = os.path.dirname(filename)
            with open(filename) as f:
                data = json.load(f)

            for cam in data['camera']:
                #print(cam)
                camParams.append([
                    [cam['world2screenMat']['e00'],cam['world2screenMat']['e01'],cam['world2screenMat']['e02'],cam['world2screenMat']['e03']],
                    [cam['world2screenMat']['e10'],cam['world2screenMat']['e11'],cam['world2screenMat']['e12'],cam['world2screenMat']['e13']],
                    [cam['world2screenMat']['e20'],cam['world2screenMat']['e21'],cam['world2screenMat']['e22'],cam['world2screenMat']['e23']],
                    [cam['world2screenMat']['e30'],cam['world2screenMat']['e31'],cam['world2screenMat']['e32'],cam['world2screenMat']['e33']],
                ])
                camPoses.append([
                    cam['pos']['x'],cam['pos']['y'],cam['pos']['z']
                ])
                silhouetteImgs.append(
                    cv2.imread(os.path.join(folder,cam['img'][2]))
                )
            mesh.loadTexturesAndMats(camParams,silhouetteImgs,camPoses)

        elif '.ply' in filename:
            mesh.loadPly(filename)

ui.actionimport.triggered.connect(importResource)

def savePannelAsImage():
    ui.openGLWidget_2.grabFramebuffer().save('./texture.png')

ui.actionscreenshot.triggered.connect(savePannelAsImage)


sys.exit(app.exec_())
