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

from args import build_argparser
args = build_argparser().parse_args()

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

scene = QtGLScene(ui.openGLWidget)

uniformModel = Uniform()

def mainloop():
    uniformModel.setValue('viewPos', [
        scene.camera.position[0],
        scene.camera.position[1],
        scene.camera.position[2]])
    scene.startDraw()
    scene.endDraw()

MainWindow.show()
timer = QTimer(MainWindow)
timer.timeout.connect(mainloop)
timer.start(1)

# projective texture model 
tex1 = Texture(np.zeros((2,2)))
depthMap = FrameBuffer(scene.size)
uniformModel.addTexture('depthMap',depthMap)
uniformModel.addTexture('projectTex',tex1)
uniformModel.addMat4('projectMat',np.identity(4))
uniformModel.addMat4('normalizeMat', np.identity(4))

import components.opengl.shaderLib.PhongShading as PhongShading
matModel = ShaderMaterial(PhongShading.vertex_shader,
                     PhongShading.fragment_shader,
                     uniformModel)

from components.opengl.BufferGeometry import *
geoModel = BufferGeometry()
model = Mesh(matModel, geoModel)
model.wireframe = False
scene.add(model)

# depth test for project texture
depthMapUniform = Uniform()
depthMapUniform.addMat4('normalizeMat', np.identity(4))
depthMapUniform.addMat4('worldToSrcreen',np.identity(4))
# camera pose test
import components.opengl.shaderLib.DepthMap as DepthMap
matdepthMapModel = ShaderMaterial(DepthMap.vertex_shader,
                     DepthMap.fragment_shader,
                     depthMapUniform)
depthMapModel = Mesh(matdepthMapModel, geoModel)

scene.startDraw()
depthMapModel.init()
scene.endDraw()
def customPaint():
    depthMap.updateResolution(scene.size)
    depthMap.startDraw()
    depthMapModel.draw()
    depthMap.endDraw()

scene.customRender.append(customPaint)

def importResource():
    qfd = QFileDialog()
    filter = "(*.json *.ply)"
    filenames,_ = QFileDialog.getOpenFileNames(qfd,'import', './', filter)

    camParams=[]
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
                silhouetteImgs.append(
                    cv2.imread(os.path.join(folder,cam['img'][2]))
                )

            index = 5
            scene.startDraw()
            silhouetteImg = Texture(silhouetteImgs[index])
            silhouetteImg.init()
            uniformModel.setValue('projectTex',silhouetteImg)
            uniformModel.setValue('projectMat',camParams[index])
            depthMapUniform.setValue('worldToSrcreen',camParams[index])
            scene.endDraw()             
 
        elif '.ply' in filename:
            scene.startDraw()
            sucess = geoModel.readObj(filename)
            if sucess:
                uniformModel.setValue('normalizeMat',geoModel.getNormalizeMat())                
            
            scene.endDraw()

ui.actionimport.triggered.connect(importResource)

def savePannelAsImage():
    ui.openGLWidget_2.grabFramebuffer().save('./texture.png')

ui.actionscreenshot.triggered.connect(savePannelAsImage)


sys.exit(app.exec_())
