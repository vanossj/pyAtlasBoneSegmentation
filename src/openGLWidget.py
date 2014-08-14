# -*- coding:utf-8 -*-
"""
Created on Aug 25, 2011

@author: grant
"""
import sys, math, numpy
from openGLUtils import Camera, OpenGlLight
from PySide import QtCore, QtGui, QtOpenGL
from OpenGL import GL, GLU
from Joint import Joint
import TriModel


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.keyStates = {}
        self.keyStates[16777249] = False
        self.camera = Camera()
        self.worldJoint = None
        self.setSizePolicy(QtGui.QSizePolicy.Policy.Expanding, QtGui.QSizePolicy.Policy.Expanding)
        self.initializeOpenGlLights()
        self.colorDrivenMaterial = None
        self.prespective = True
        self.useCallLists = True

    def clearScene(self):
        if self.worldJoint is not None:
            # remove current scene
            self.worldJoint.childJoints = []
            for model in self.worldJoint.models:
                if model.name[:5] != 'axis_':
                    self.worldJoint.models.remove(model)

    def setColorDrivenMaterial(self, enabled=False):
        self.colorDrivenMaterial = enabled
        if enabled:
            GL.glEnable(GL.GL_COLOR_MATERIAL)
        else:
            GL.glDisable(GL.GL_COLOR_MATERIAL)

    def initializeOpenGlLights(self):
        self.lights = []
        for i in xrange(8):
            self.lights.append(OpenGlLight(i))
        self.lights[0].enabled = True
        self.lights[0].diffuseColor = [1.0, 1.0, 1.0, 1.0]
        self.lights[0].specularColor = [1.0, 1.0, 1.0, 1.0]
        self.lights[0].position = [1.0, 0.5, 1.0, 0.0]

    def minimumSizeHint(self):
        # inherited from QGLWidget, sets minimum size
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        # inherited from QGLWidget, sets start size
        return QtCore.QSize(1600, 1600)

    def initializeGL(self):
        self.worldJoint = Joint(name='WorldJoint', showAxis=True, axisScale=0.7)

        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
#        GL.glClearColor(1.0,1.0,1.0,1.0)
        GL.glClearDepth(1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glEnable(GL.GL_BLEND)  # enable blending
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)  # set blend function
        GL.glEnable (GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)  # enable the lighting
#        GL.glEnable(GL.GL_LIGHT0) #enable LIGHT0, our Diffuse Light
        for light in self.lights:
            light.updateOpenGl()
        GL.glShadeModel(GL.GL_SMOOTH)  # set the shader to smooth shader
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glEnable(GL.GL_TEXTURE_2D)
        vMin, vMax = self.getBoundingBox()
        self.camera.setView(vMin, vMax, view='ortho1')

    def getBoundingBox(self):
        vMin = None
        vMax = None

        if self.worldJoint is not None:
            vMin, vMax = self.worldJoint.getBoundingBox()

        if vMin is None:
            vMin = numpy.array([0.0, 0.0, 0.0])
        if vMax is None:
            vMax = numpy.array([1.0, 1.0, 1.0])
        return vMin, vMax

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        self.camera.openGLTransform()
        if self.worldJoint is not None:
            self.worldJoint.OpenGLPaint(self.colorDrivenMaterial, self.useCallLists)

    def resizeGL(self, width, height):
        if width == 0 or height == 0:
            return
        # Set our viewport to the size of our window
        GL.glViewport(0, 0, width, height)
        # Switch to the projection matrix so that we can manipulate how our scene is viewed
        GL.glMatrixMode(GL.GL_PROJECTION)
        # Reset the projection matrix to the identity matrix so that we don't get any artifacts (cleaning up)
        GL.glLoadIdentity()
        # Set the Field of view angle (in degrees), the aspect ratio of our window, and the new and far planes
        vMin, vMax = self.getBoundingBox()
        near = 1.0
        far = 100.0
        aspectRatio = float(width) / float(height)
        if vMax is not None and vMin is not None:
            d = vMax - vMin
            d = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            far = d * 3.0
#        print 'Far plane: ',far
        if self.prespective:
            GLU.gluPerspective(60, aspectRatio, near, far)
        else:
            h = (vMax[1] - vMin[1]) * aspectRatio
            center = (vMax[1] - vMin[1]) / 2
            GL.glOrtho(center - h / 2, center + h / 2, vMin[1], vMax[1], near, far)

        # Switch back to the model view matrix, so that we can start drawing shapes correctly
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def keyPressEvent(self, event):
        global camAngle
        key = event.key()
        txt = event.text()
        self.keyStates[key] = True
        if key == 16777235:  # Up
            self.camera.rotateAroundAxis(math.pi / 180.0, [1.0, 0.0, 0.0])
        elif key == 16777237:  # Down
            self.camera.rotateAroundAxis(-math.pi / 180.0, [1.0, 0.0, 0.0])
        elif key == 16777234:  # Left
            self.camera.rotateAroundAxis(math.pi / 180.0, [0.0, 1.0, 0.0])
        elif key == 16777236:  # Right
            self.camera.rotateAroundAxis(-math.pi / 180.0, [0.0, 1.0, 0.0])
        elif key == 16777249:  # crtl
            pass
        elif txt == 's':
            pass
        else:
#            print key, event.text()
            pass

        # force redraw of scene
        self.updateGL()

    def keyReleaseEvent(self, event):
        key = event.key()
        self.keyStates[key] = False

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        if (event.buttons() & QtCore.Qt.LeftButton) and (self.keyStates[16777249] is False):
            self.camera.rotateAroundAxis(math.radians(-1 * dy), [1, 0, 0])
            self.camera.rotateAroundAxis(math.radians(-1 * dx), [0, 1, 0])
            # force redraw of scene
            self.updateGL()
        elif (event.buttons() & QtCore.Qt.MidButton) or ((event.buttons() & QtCore.Qt.LeftButton) and (self.keyStates[16777249])):
            # Pan if middle mouse button is pressed or left button is pressed and control is pressed
            # screen height in units is same as distance from camLoc to camFocus if view angle is 60 degrees
            unitsPerPixel = numpy.linalg.norm(self.camera.camLoc - self.camera.camFocus) / self.height()
            self.camera.pan([-dx * unitsPerPixel, dy * unitsPerPixel, 0.0])
            # force redraw of scene
            self.updateGL()
        self.lastPos = event.pos()

    def wheelEvent(self, event):
        d = 1.0 + 0.1 * abs(event.delta()) / 120.0
        if event.delta() > 0.0:
            self.camera.zoom(1 / d)
        else:
            self.camera.zoom(d)
        # force redraw of scene
        self.updateGL()


class Window(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.glWidget = GLWidget()
        createDefaultScene(self.glWidget.worldJoint)
        mainLayout = QtGui.QHBoxLayout()
        mainLayout.addWidget(self.glWidget)
        self.setLayout(mainLayout)
        self.setWindowTitle(self.tr("Qt openGL Viewer"))


def createDefaultScene(worldJoint):
    # create and connect joints
    joint1_2 = Joint([0.5, 0.5, 3.5], parentJoint=worldJoint, showAxis=True, axisScale=0.5)
    joint2_3 = Joint([0.5, 0.5, 7.5], parentJoint=joint1_2, showAxis=True, axisScale=0.5)

    # create bone models
    TriModel.createRectangularSolid([1, 1, 3], [0, 0, 0], joint=worldJoint)
    TriModel.createRectangularSolid([1, 1, 3], [0, 0, 4], joint=joint1_2)
    TriModel.createRectangularSolid([1, 1, 3], [0, 0, 8], joint=joint2_3)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
