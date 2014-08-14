'''
Created on Jul 27, 2011

@author: grant

Ideas for Improvement
1.    Create Ball&Socket joint class as well as Hinge joint class that inherit
    from joint class, these classes will call a specific version of
    joint.rotate(). The ball&socket joint is very similar to spherical except
    the vector that the model spins around can be arbitrary (spherical spin
    vector is [0,0,1]. Relevant methods of moving a joint should be selectable
    from combobox on GUI, i.e. a Hinge joint could only move in hinge mode, but
    a joint could move in rectangular, spherical, hinge, or ball&socket mode.

2.    Different Coordinate systems should have different axis models. Rotation
    in rectangular system should display different axis then a rotation in
    spherical system. Similarly a Hinge joint and a Ball&Socket joint should
    have unique axis models

4.    Fix determining extends box so it only has to do that once, then saves the
    values (maybe, might not be a good idea if model size could change)

5.    Set up orthographic projection view

6.    Setup Atlas so that default joint orientation can be set. Also joint types
    should be setable in the atlas xml

'''
import sys, math, time
import numpy
from scipy.ndimage.filters import gaussian_filter
import TriModel
from Joint import Joint
from PySide import QtCore, QtGui
from OpenGLGUI import Ui_MainWindow
from openGLWidget import GLWidget
from STL import readSTLfile, saveSTLFile
from lxml import etree
import os
import os.path
from BinVoxReader import BinVox
from polygonise import Polygonise
import dicom
import AlignAtlas
from InputDataTypes import UserDicom, UserAtlas
import numpyTransform
import InputDataTypes
import scipy.stats
from scipy.io.matlab.mio import savemat
from scipy.spatial import cKDTree
from cgkit.cgtypes import quat
from OpenGL import GL


class MyMainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        self.lastStep = 0

        super(MyMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setTabPosition(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas, QtGui.QTabWidget.TabPosition.North)
        # Set up Docks in a tab format
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.ui.dockJoint)
        self.tabifyDockWidget(self.ui.dockJoint, self.ui.dockModel)
        self.tabifyDockWidget(self.ui.dockJoint, self.ui.dockSlice)
        self.tabifyDockWidget(self.ui.dockJoint, self.ui.dockLights)
        self.tabifyDockWidget(self.ui.dockJoint, self.ui.dockCamera)
        self.tabifyDockWidget(self.ui.dockJoint, self.ui.dockScene)
        self.tabifyDockWidget(self.ui.dockJoint, self.ui.dockTol)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.ui.dockItemList)

        for child in self.children():
            if isinstance(child, QtGui.QTabBar) and child.count() == 7:  # make sure proper number of tabs
                child.setCurrentIndex(0)

        self.glWidget = GLWidget()
        self.setupSceneControls()

        self.ui.openGlTab.layout().addWidget(self.glWidget)
        QtCore.QObject.connect(self.ui.saveCSV, QtCore.SIGNAL("clicked()"), self.saveCSV)
        QtCore.QObject.connect(self.ui.copyTable, QtCore.SIGNAL("clicked()"), self.copyTable)
        QtCore.QObject.connect(self.ui.chooseSpecularColorCam, QtCore.SIGNAL("clicked()"), self.setSpecularColorCam)
        QtCore.QObject.connect(self.ui.chooseDiffuseColorCam, QtCore.SIGNAL("clicked()"), self.setDiffuseColorCam)
        QtCore.QObject.connect(self.ui.chooseAmbientColorCam, QtCore.SIGNAL("clicked()"), self.setAmbientColorCam)
        QtCore.QObject.connect(self.ui.chooseEmissiveColorCam, QtCore.SIGNAL("clicked()"), self.setEmissiveColorCam)
        QtCore.QObject.connect(self.ui.chooseSpecularColorMat, QtCore.SIGNAL("clicked()"), self.setSpecularColorMat)
        QtCore.QObject.connect(self.ui.chooseDiffuseColorMat, QtCore.SIGNAL("clicked()"), self.setDiffuseColorMat)
        QtCore.QObject.connect(self.ui.chooseAmbientColorMat, QtCore.SIGNAL("clicked()"), self.setAmbientColorMat)
        QtCore.QObject.connect(self.ui.chooseEmissiveColorMat, QtCore.SIGNAL("clicked()"), self.setEmissiveColorMat)
        QtCore.QObject.connect(self.ui.matShinySlider, QtCore.SIGNAL("valueChanged(int)"), self.setShininessColorMat)
        QtCore.QObject.connect(self.ui.matShinySpinBox, QtCore.SIGNAL("valueChanged(double)"), self.setShininessColorMat)
        QtCore.QObject.connect(self.ui.modelVisible, QtCore.SIGNAL("stateChanged(int)"), self.setModelVisibility)

        QtCore.QObject.connect(self.ui.setView, QtCore.SIGNAL("currentIndexChanged(QString)"), self.setView)
        QtCore.QObject.connect(self.ui.lightNum, QtCore.SIGNAL("valueChanged(int)"), self.updateLightPropertiesTab)
        QtCore.QObject.connect(self.ui.lightXPos, QtCore.SIGNAL("valueChanged(double)"), self.setLightXPos)
        QtCore.QObject.connect(self.ui.lightYPos, QtCore.SIGNAL("valueChanged(double)"), self.setLightYPos)
        QtCore.QObject.connect(self.ui.lightZPos, QtCore.SIGNAL("valueChanged(double)"), self.setLightZPos)
        QtCore.QObject.connect(self.ui.lightEnable, QtCore.SIGNAL("stateChanged(int)"), self.setLightEnable)
        QtCore.QObject.connect(self.ui.directionalLight, QtCore.SIGNAL("stateChanged(int)"), self.setLightDirectional)
        QtCore.QObject.connect(self.ui.itemTreeList, QtCore.SIGNAL("itemSelectionChanged()"), self.itemChangedCB)
        QtCore.QObject.connect(self.ui.jointXRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateJointSlider)
        QtCore.QObject.connect(self.ui.jointYRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateJointSlider)
        QtCore.QObject.connect(self.ui.jointZRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateJointSlider)
        QtCore.QObject.connect(self.ui.jointXRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateJointSpinBox)
        QtCore.QObject.connect(self.ui.jointYRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateJointSpinBox)
        QtCore.QObject.connect(self.ui.jointZRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateJointSpinBox)
        QtCore.QObject.connect(self.ui.rotationOrder, QtCore.SIGNAL("currentIndexChanged(QString)"), self.rotationOrderChanged)
        QtCore.QObject.connect(self.ui.sphericalCoord, QtCore.SIGNAL("stateChanged(int)"), self.setRotateCoord)
        QtCore.QObject.connect(self.ui.enableColorDrivenModelsCheckBox, QtCore.SIGNAL("stateChanged(int)"), self.setColorModel)
        QtCore.QObject.connect(self.ui.pointSize, QtCore.SIGNAL("valueChanged(double)"), self.pointSize)

        # callbacks for slice view changes
        QtCore.QObject.connect(self.ui.spinBox100AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice100Axis)
        QtCore.QObject.connect(self.ui.slider100AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice100Axis)
        QtCore.QObject.connect(self.ui.spinBox010AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice010Axis)
        QtCore.QObject.connect(self.ui.slider010AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice010Axis)
        QtCore.QObject.connect(self.ui.spinBox001AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice001Axis)
        QtCore.QObject.connect(self.ui.slider001AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice001Axis)

        QtCore.QObject.connect(self.ui.GrabScreen, QtCore.SIGNAL("clicked()"), self.grabScreen)
        QtCore.QObject.connect(self.ui.automate, QtCore.SIGNAL("clicked()"), self.automate)
        QtCore.QObject.connect(self.ui.camUpdate, QtCore.SIGNAL("clicked()"), self.camUpdate)
        QtCore.QObject.connect(self.ui.invertNormals, QtCore.SIGNAL("clicked()"), self.invertNormals)
        QtCore.QObject.connect(self.ui.saveModel, QtCore.SIGNAL("clicked()"), self.saveModel)
        QtCore.QObject.connect(self.ui.loadModel, QtCore.SIGNAL("clicked()"), self.loadModel)
        QtCore.QObject.connect(self.ui.loadAtlas, QtCore.SIGNAL("clicked()"), self.loadAtlas)
        QtCore.QObject.connect(self.ui.defaultScene, QtCore.SIGNAL("clicked()"), self.createDefaultScene)
        QtCore.QObject.connect(self.ui.clearScene, QtCore.SIGNAL("clicked()"), self.clearScene)
        QtCore.QObject.connect(self.ui.projection, QtCore.SIGNAL("currentIndexChanged(int)"), self.setProjection)
        QtCore.QObject.connect(self.ui.useCallLists, QtCore.SIGNAL("stateChanged(int)"), self.setUseCallLists)
        QtCore.QObject.connect(self.ui.jointScaleX, QtCore.SIGNAL("valueChanged(double)"), self.setJointScale)
        QtCore.QObject.connect(self.ui.jointScaleY, QtCore.SIGNAL("valueChanged(double)"), self.setJointScale)
        QtCore.QObject.connect(self.ui.jointScaleZ, QtCore.SIGNAL("valueChanged(double)"), self.setJointScale)
        QtCore.QObject.connect(self.ui.jointScaleXYZ, QtCore.SIGNAL("valueChanged(double)"), self.setJointScaleIsometric)
        QtCore.QObject.connect(self.ui.jointName, QtCore.SIGNAL("editingFinished()"), self.setJointName)
        QtCore.QObject.connect(self.ui.modelName, QtCore.SIGNAL("editingFinished()"), self.setModelName)
        QtCore.QObject.connect(self.ui.jointXPosStep, QtCore.SIGNAL("clicked()"), self.moveJointX)
        QtCore.QObject.connect(self.ui.jointYPosStep, QtCore.SIGNAL("clicked()"), self.moveJointY)
        QtCore.QObject.connect(self.ui.jointZPosStep, QtCore.SIGNAL("clicked()"), self.moveJointZ)
        QtCore.QObject.connect(self.ui.modelAlpha, QtCore.SIGNAL("valueChanged(int)"), self.setModelAlpha)

        # tolerence vector callbacks
        QtCore.QObject.connect(self.ui.tolXRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateTolSlider)
        QtCore.QObject.connect(self.ui.tolYRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateTolSlider)
        QtCore.QObject.connect(self.ui.tolZRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateTolSlider)
        QtCore.QObject.connect(self.ui.tolXRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateTolSpinBox)
        QtCore.QObject.connect(self.ui.tolYRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateTolSpinBox)
        QtCore.QObject.connect(self.ui.tolZRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateTolSpinBox)
        QtCore.QObject.connect(self.ui.tolVecAngle, QtCore.SIGNAL("valueChanged(double)"), self.makeTolVec)
        QtCore.QObject.connect(self.ui.tolVecLength, QtCore.SIGNAL("valueChanged(double)"), self.updateTolVec)
        QtCore.QObject.connect(self.ui.tolVecXOrigin, QtCore.SIGNAL("valueChanged(double)"), self.updateTolVec)
        QtCore.QObject.connect(self.ui.tolVecYOrigin, QtCore.SIGNAL("valueChanged(double)"), self.updateTolVec)
        QtCore.QObject.connect(self.ui.tolVecZOrigin, QtCore.SIGNAL("valueChanged(double)"), self.updateTolVec)

        # link keypress event for nuclear medicine table
        self.ui.nmTable.keyPressEvent = self.nmTableKeyPressEvent

        self.updateLightPropertiesTab(self.ui.lightNum.value())
        self.setColorModel(self.ui.enableColorDrivenModelsCheckBox.checkState())
        # configure color mode
        if self.ui.enableColorDrivenModelsCheckBox.checkState() == QtCore.Qt.CheckState.Checked:
            self.glWidget.setColorDrivenMaterial(True)
        else:
            self.glWidget.setColorDrivenMaterial(False)
        self.selectedJoint = None
        self.selectedModel = None
        self.referenceModelJoint = None
        self.atlasJoint = None
        self.tolJoint = None
        self.atlasAxes = numpy.identity(3)  # TODO: this should really be in an Atlas class
        self.scanData = None

    def pointSize(self, size):
#        print 'Previous Point Size', GL.glGetDouble(GL.GL_POINT_SIZE)
        if self.selectedModel is not None:
            self.selectedModel.pointSize = size
        self.glWidget.updateGL()

    def nmTableKeyPressEvent(self, event):
        key = event.key()
        modifer = event.modifiers()

        if key == 67 and modifer == QtCore.Qt.ControlModifier:  # CTRL + c
            self.copyTable()

    def tableTostring(self, rowDelimiter='\n', columnDelimiter='\t', wholeTable=False):
        selectedRanges = self.ui.nmTable.selectedRanges()
        if len(selectedRanges) == 0 or wholeTable is True:
            left = 0
            right = self.ui.nmTable.columnCount()
            top = 0
            bottom = self.ui.nmTable.rowCount()
        else:
            left = selectedRanges[0].leftColumn()
            right = selectedRanges[0].rightColumn() + 1
            top = selectedRanges[0].topRow()
            bottom = selectedRanges[0].bottomRow() + 1
        text = 'Region' + columnDelimiter
        # create column headers
        for j in xrange(left, right):
            text += self.ui.nmTable.horizontalHeaderItem(j).text() + columnDelimiter
        text += rowDelimiter

        # create data entries
        for i in xrange(top, bottom):
            text += self.ui.nmTable.verticalHeaderItem(i).text() + columnDelimiter
            for j in xrange(left, right):
                text += self.ui.nmTable.item(i, j).text() + columnDelimiter
            text += rowDelimiter
        return text

    def saveCSV(self):
        filepath, selectedFilter = QtGui.QFileDialog.getSaveFileName(caption="Save Nuclear Medicine Table As", filter="CSV (*.csv);;Any File (*.*)")  # @UnusedVariable
        if len(filepath) == 0:
            return
        f = file(filepath, 'w')
        f.write(self.tableTostring(columnDelimiter=',', wholeTable=True))
        f.close()

    def copyTable(self):
        QtGui.QClipboard().setText(self.tableTostring(wholeTable=True))

    def setModelAlpha(self, alpha, model=None):
        if model is None:
            model = self.selectedModel
        if model is not None:
            if len(model.specularColor) == 3:
                model.specularColor.append(alpha / 255.0)
            elif len(model.specularColor) > 3:
                model.specularColor[3] = alpha / 255.0
            if len(model.ambientColor) == 3:
                model.specularColor.append(alpha / 255.0)
            elif len(model.ambientColor) > 3:
                model.ambientColor[3] = alpha / 255.0
            if len(model.diffuseColor) == 3:
                model.diffuseColor.append(alpha / 255.0)
            elif len(model.diffuseColor) > 3:
                model.diffuseColor[3] = alpha / 255.0
            if len(model.emissionColor) == 3:
                model.emissionColor.append(alpha / 255.0)
            elif len(model.emissionColor) > 3:
                model.emissionColor[3] = alpha / 255.0
            self.glWidget.updateGL()

    def setupSceneControls(self):
        '''function that populates/resets all scene controls'''
        self.populateItemTree(self.glWidget.worldJoint)
        self.selectedJoint = None
        self.selectedModel = None
        self.ui.jointName.setText('No Joint Selected')
        self.ui.modelName.setText('No Model Selected')

    def setUseCallLists(self, state):
        if state == QtCore.Qt.CheckState.Checked:
            self.glWidget.useCallLists = True
        else:
            self.glWidget.useCallLists = False

    def setJointScale(self, scale):
        if self.selectedJoint is not None:
            self.selectedJoint.scale(self.ui.jointScaleX.value(), self.ui.jointScaleY.value(), self.ui.jointScaleZ.value())
            self.glWidget.updateGL()

    def setJointScaleIsometric(self, scale):
        if self.selectedJoint is not None:
            self.selectedJoint.scale(self.ui.jointScaleXYZ.value())
            self.glWidget.updateGL()

    def setProjection(self, i):
        if i == 0:
            self.glWidget.prespective = True
        elif i == 1:
            self.glWidget.prespective = False
        self.glWidget.resizeGL(self.glWidget.width(), self.glWidget.height())

    def camUpdate(self):
        self.ui.camXPos.setValue(self.glWidget.camera.camLoc[0])
        self.ui.camYPos.setValue(self.glWidget.camera.camLoc[1])
        self.ui.camZPos.setValue(self.glWidget.camera.camLoc[2])
        self.ui.camXLookAt.setValue(self.glWidget.camera.camFocus[0])
        self.ui.camYLookAt.setValue(self.glWidget.camera.camFocus[1])
        self.ui.camZLookAt.setValue(self.glWidget.camera.camFocus[2])
        self.ui.camXUp.setValue(self.glWidget.camera.camUp[0])
        self.ui.camYUp.setValue(self.glWidget.camera.camUp[1])
        self.ui.camZUp.setValue(self.glWidget.camera.camUp[2])
        self.ui.camQuatW.setValue(self.glWidget.camera.cameraQuat.w)
        self.ui.camQuatX.setValue(self.glWidget.camera.cameraQuat.x)
        self.ui.camQuatY.setValue(self.glWidget.camera.cameraQuat.y)
        self.ui.camQuatZ.setValue(self.glWidget.camera.cameraQuat.z)
        self.ui.camFunction.setText('self.glWidget.camera.setView(quat(%0.10f,%0.10f,%0.10f,%0.10f),[%0.10f,%0.10f,%0.10f],[%0.10f,%0.10f,%0.10f])' % (self.glWidget.camera.cameraQuat.w, self.glWidget.camera.cameraQuat.x, self.glWidget.camera.cameraQuat.y, self.glWidget.camera.cameraQuat.z, self.glWidget.camera.camLoc[0], self.glWidget.camera.camLoc[1], self.glWidget.camera.camLoc[2], self.glWidget.camera.camFocus[0], self.glWidget.camera.camFocus[1], self.glWidget.camera.camFocus[2]))

    def setJointName(self):
        if self.selectedJoint is not None:
            self.selectedJoint.name = self.ui.jointName.text()
            self.populateItemTree(self.glWidget.worldJoint)

    def setModelName(self):
        if self.selectedModel is not None:
            self.selectedModel.name = self.ui.modelName.text()
            self.populateItemTree(self.glWidget.worldJoint)

    def invertNormals(self):
        self.selectedModel.invertNormals()
        self.glWidget.updateGL()

    def saveModel(self):
        if self.selectedModel is not None:
            filePath, filtername = QtGui.QFileDialog.getSaveFileName(caption="Save Model as...", filter="STL ASCII file (*.stl);;STL Binary file (*.stl)")
            if filtername == "STL Binary file (*.stl)":
                saveSTLFile(self.selectedModel, filePath)
            elif filtername == "STL ASCII file (*.stl)":
                saveSTLFile(self.selectedModel, filePath, False)

    def grabScreen(self):
#        GL.glClearColor(1.0,1.0,1.0,1.0)
#        self.glWidget.updateGL()
#        time.sleep(0.5)

        pic = self.glWidget.grabFrameBuffer()
        QtGui.QClipboard().setImage(pic)

#        pic = self.glWidget.renderPixmap()
#        QtGui.QClipboard().setPixmap(pic)

#        GL.glClearColor(0.0,0.0,0.0,1.0)
#        self.glWidget.updateGL()

    def automate(self):
        savePrefix = None
        savePrefix = '1-10_'
        start = time.time()

        # load data
        t1 = time.time()
        self.atlasData = UserAtlas('atlases\MOBY_package\MOBY Atlas.xml', self.glWidget.worldJoint)
        print 'Time to Open Atlas:', time.time() - t1
        t1 = time.time()

        if self.scanData is None:
        	ctFilepath, _flter = QtGui.QFileDialog.getOpenFileName(caption="Select a CT dataset to load", filter="DICOM (*.dcm)")
        	if len(ctFilepath) == 0:
        		return
    		niFilepath, _flter = QtGui.QFileDialog.getOpenFileName(caption="Select a matching NI dataset to load", filter="DICOM (*.dcm)")
        	if len(niFilepath) == 0:
        		return
        # self.scanData = UserDicom(r'dicom\test\TEST_1-10_CT.dcm', r'dicom\test\TEST_1-10_SPECT.dcm',
        #                           joint=self.glWidget.worldJoint, sigma=2, resampleFactor=3, isolevel=350.0)
		self.scanData = UserDicom(ctFilepath, niFilepath, joint=self.glWidget.worldJoint, sigma=2, resampleFactor=3, isolevel=350.0)

        print 'Time to Open Dicom files:', time.time() - t1

        self.scanData.isosurfaceModel.visible = False
        self.setModelAlpha(0.7 * 255, self.scanData.isosurfaceModel1)
        self.setModelAlpha(0.7 * 255, self.scanData.isosurfaceModel2)
        self.setModelAlpha(0.7 * 255, self.scanData.isosurfaceModel3)

        # set camera and redraw scene
        self.glWidget.camera.setView(quat(-0.1450718126, -0.4014994641, -0.1638632680, 0.8893262500), [-681.1013854165, -353.9837864636, 1403.4630773443], [196.4125000000, 183.0531656115, 582.1479125977])
        self.setupSceneControls()
        self.glWidget.resizeGL(self.glWidget.width(), self.glWidget.height())
        self.glWidget.updateGL()

        # set up slice view controls
        self.setSlice100Axis(0)
        self.setSlice010Axis(0)
        self.setSlice001Axis(0)
        self.ui.spinBox001AxisSlice.setMaximum(self.scanData.ctField.shape[2])
        self.ui.slider001AxisSlice.setMaximum(self.scanData.ctField.shape[2])
        self.ui.spinBox010AxisSlice.setMaximum(self.scanData.ctField.shape[1])
        self.ui.slider010AxisSlice.setMaximum(self.scanData.ctField.shape[1])
        self.ui.spinBox100AxisSlice.setMaximum(self.scanData.ctField.shape[0])
        self.ui.slider100AxisSlice.setMaximum(self.scanData.ctField.shape[0])

        # Rough Align Atlas to skeleton
        alignmentAxis, neckPos, scaleFactor, hipLoc, spineModel, tailModel, spinePCA = AlignAtlas.roughAlign(self.scanData, self.atlasData , verbose=True, visualize=False)
		# alignmentAxis, neckPos, scaleFactor, hipLoc, spineModel, tailModel, spinePCA = AlignAtlas.roughAlign(self.scanData, self.atlasData , verbose=True, visualize=False,
  #                                           spineModelShortcut=savePrefix + 'SpineModel.stl', tailModelShortcut=savePrefix + 'TailModel.stl')

        print 'Rough align Axis:'
        print alignmentAxis

        # add spine and tail models to scene
        # TODO: spine and tail models should be a replacement for the spine in the atlas, find a way to add it to the atlas
        spineModel.setJoint(self.glWidget.worldJoint)
        tailModel.setJoint(self.glWidget.worldJoint)

        # roughAlign() returns rotation array of vectors [anteriorVector, dorsalVector, rightVector]
        # generate a rotation matrix that will transform the atlas Axes to the CT axes
        rotationMat = numpyTransform.coordinateSystemConversionMatrix(self.atlasData.atlasAxes, alignmentAxis, N=4)
        print 'Function alignment Axis:'
        print rotationMat

        for childJoint in self.atlasData.atlasJoint.childJoints:
            if childJoint.name == 'Neck':
                break
        offset = neckPos - childJoint.locationUnityScale
        offset = numpy.squeeze(numpy.array(offset))
        print 'Offset:', offset
        print 'Atlas Neck Position:', childJoint.locationUnityScale
        print 'Dicom Neck Position:', neckPos

#        #manually adjust scale factor
#        scaleFactor *= 1.35

        self.atlasData.atlasJoint.scale(scaleFactor)
        self.atlasData.atlasJoint.rotate(rotationMat)
        self.atlasData.atlasJoint.translate(offset, absolute=False)

        print 'Scale Matrix:'
        print self.atlasData.atlasJoint.scaleMat
        print 'Rotation Matrix:'
        print self.atlasData.atlasJoint.rotateMat
        print 'Offset Matrix:'
        print self.atlasData.atlasJoint.translateMat
        print 'Initial Offset Matrix'
        print self.atlasData.atlasJoint.initialLocationMat

        # hide not registered atlas bones
        self.atlasData.atlasModels['Spine'].visible = False
        self.atlasData.atlasModels['Scapula Right'].visible = False
        self.atlasData.atlasModels['Scapula Left'].visible = False
        self.atlasData.atlasModels['Upper Forelimb Right'].visible = False
        self.atlasData.atlasModels['Upper Forelimb Left'].visible = False
        self.atlasData.atlasModels['Lower Forelimb Right'].visible = False
        self.atlasData.atlasModels['Lower Forelimb Left'].visible = False
        self.atlasData.atlasModels['Forepaw Right'].visible = False
        self.atlasData.atlasModels['Forepaw Left'].visible = False

        # set camera and redraw scene
        self.glWidget.camera.setView(quat(0.3381642550, 0.9222312229, -0.0475532999, -0.1813096573), [79.9461811889, -88.6397062133, -115.5570538501], [188.3407945769, 90.6932903010, 93.0659155950])
        self.setupSceneControls()
        self.glWidget.resizeGL(self.glWidget.width(), self.glWidget.height())
        self.glWidget.updateGL()

        # manually create kdtree for model
        self.scanData.isosurfaceModel1.kdtree = cKDTree(self.scanData.isosurfaceModel1.transformedVertexList * self.scanData.resampleFactor)

#        #Get data for paper
#        self.atlasData.atlasJoint.transformVertices()
#        dist = 0.0
#        numPoints = 0.0
#        for bone in self.scanData.bonesVol1:
#            print 'Post Rough Align %s Joint Location %s' % (self.atlasData.atlasModels[bone].joint.name, self.atlasData.atlasModels[bone].joint.location)
#            d, i = self.scanData.isosurfaceModel1.kdtree.query(self.atlasData.atlasModels[bone].transformedVertexList)
#            dist += numpy.sum(d)
#            numPoints += len(i)
#            print 'Post Rough Align Mean distance from %s to surface: %f' % (self.atlasData.atlasModels[bone].name, numpy.sum(d)/len(i))
#        print 'Post Rough Align Mean distance all bones to surface: %f' % (dist/numPoints)


#        #Create reference object, this is useful for checking position and orientation of other models
#        scale=1.0
#        size=scale*numpy.array([1.0,1.0,1.0])
#        offset2 = scale*numpy.array([-0.5,-0.5,-0.5])
#        refPointJoint1=Joint(parentJoint=self.glWidget.worldJoint, name='Ref Point 1 Joint')
#        TriModel.createRectangularSolid(size,offset2, joint=refPointJoint1, name='Ref Point',color=[1.0,0.0,0.0,1.0])
#
#        #Create reference object
#        scale=1.0
#        size=scale*numpy.array([1.0,1.0,1.0])
#        offset2 = scale*numpy.array([-0.5,-0.5,-0.5])
#        refPointJoint2=Joint(parentJoint=self.glWidget.worldJoint, name='Ref Point 2 Joint')
#        TriModel.createRectangularSolid(size,offset2, joint=refPointJoint2, name='Ref Point',color=[0.0,1.0,0.0,1.0])

        # Do Fine align
        AlignAtlas.FineAlign(self.scanData, self.atlasData, hipLocation=hipLoc, spineVertices=spineModel.OriginalVertexList,
                             tailVertices=tailModel.OriginalVertexList,
                             getResults=False, updateSceneFunc=self.glWidget.updateGL)

        # return

        # Create NM data sized mask of CT data
        boneVertexLists = []
        self.atlasData.atlasJoint.transformVertices()
        skullVlist = numpy.append(self.atlasData.atlasModels['Skull Inside'].transformedVertexList, self.atlasData.atlasModels['Skull Outside'].transformedVertexList)
        skullVlist = skullVlist.reshape((-1, 3))
        boneVertexLists.append(skullVlist)
        boneVertexLists.append(self.atlasData.atlasModels['Pelvis Right'].transformedVertexList)
        boneVertexLists.append(self.atlasData.atlasModels['Pelvis Left'].transformedVertexList)
        boneVertexLists.append(self.atlasData.atlasModels['Upper Hindlimb Right'].transformedVertexList)
        boneVertexLists.append(self.atlasData.atlasModels['Upper Hindlimb Left'].transformedVertexList)
        boneVertexLists.append(self.atlasData.atlasModels['Lower Hindlimb Right'].transformedVertexList)
        boneVertexLists.append(self.atlasData.atlasModels['Lower Hindlimb Left'].transformedVertexList)
        boneVertexLists.append(self.atlasData.atlasModels['HindPaw Right'].transformedVertexList)
        boneVertexLists.append(self.atlasData.atlasModels['HindPaw Left'].transformedVertexList)
        boneVertexLists.append(spineModel.transformedVertexList)
        boneVertexLists.append(tailModel.transformedVertexList)
        # the problem is that the scale and origins are off. atlas data must be scaled to match mesh units, also mask[0,0,0] is not the same as the atlas origin, this is because the mask volume had to be changed so that its size was the same as the NM data
        for i in xrange(len(boneVertexLists)):
            vlist = boneVertexLists[i]
            vlist *= self.scanData.ctDCM.SliceThickness / self.scanData.nmDCM.SliceThickness  # convert vertex points in CT units to NM units
            vlist -= self.scanData.ctVolumeNMmaskOriginLocation / self.scanData.nmDCM.SliceThickness  # subtract offset that has been scaled from mm to NM units
            boneVertexLists[i] = vlist
        mask = self.scanData.createNM_Mask(boneVertexLists)

        if savePrefix is not None:
            savemat(savePrefix + 'Mask.mat', {'Mask': mask})

#        import visvis as vv
#        app = vv.use()
#        vv.figure()
#        vv.volshow(mask)
#        app.Run()

        boneNames = ['Skull', 'Pelvis Right', 'Pelvis Left', 'Femur Right', 'Femur Left', 'Tibia Right', 'Tibia Left', 'Hindpaw Right', 'Hindpaw Left', 'Spine', 'Tail']
        self.populateNMTable(self.scanData.nmField, mask, boneNames)

#        #create texture map
#        t1 = time.time()
#        textureVolume = numpy.zeros((labelVolume.shape[0],labelVolume.shape[1],labelVolume.shape[2],3), dtype=numpy.ubyte)
#        colormap = [    [0xFF, 0x00, 0x00],
#                        [0xFF, 0x80, 0x00],
#                        [0xFF, 0xFF, 0x00],
#                        [0x80, 0xFF, 0x00],
#                        [0x00, 0xFF, 0x00],
#                        [0x00, 0xFF, 0x80],
#                        [0x00, 0xFF, 0xFF],
#                        [0x00, 0x80, 0xFF],
#                        [0x00, 0x00, 0xFF],
#                        [0x7F, 0x00, 0xFF],
#                        [0xFF, 0x00, 0xFF],
#                        [0xFF, 0x00, 0x7F],
#                        [0x80, 0x80, 0x80],
#                        [0xFF, 0x66, 0x66],
#                        [0xFF, 0xB2, 0x66],
#                        [0xFF, 0xFF, 0x66],
#                        [0xB2, 0xFF, 0x66],
#                        [0x66, 0xFF, 0x66],
#                        [0x66, 0xFF, 0xB2],
#                        [0x66, 0xFF, 0xFF],
#                        [0x66, 0xB2, 0xFF],
#                        [0x66, 0x66, 0xFF],
#                        [0xB2, 0x66, 0xFF],
#                        [0xFF, 0x66, 0xFF],
#                        [0xFF, 0x66, 0xB2],
#                        [0xC0, 0xC0, 0xC0],
#                        [0xFF, 0xFF, 0xFF]]
#        for i in xrange(uniqueValues.shape[0]):
#            if i < len(colormap):
#                textureVolume[labelVolume==uniqueValues[i]]=i
#        self.scanData.createLabelVolumeTextures(textureVolume)
#        self.scanData.setTextureToLabeledVolume()
#        print 'Time to create textures from labeled volume and apply them:',  time.time()-t1

        elapsedTime = time.time() - start
        print 'Time to Automate: %02d:%02d:%06.3f' % (int(elapsedTime / 3600), (int(elapsedTime) % 3600) / 60, elapsedTime % 60)

        # redraw gui stuff
        self.setupSceneControls()
#        self.setView()
#        self.glWidget.resizeGL(self.glWidget.width(), self.glWidget.height())
        self.glWidget.updateGL()

#        for model in self.atlasData.atlasModels:
#            print model,':',self.atlasData.atlasModels[model].joint.location

    def populateNMTable(self, nmField, labelVolume, boneNames):
        '''
        The index of the bone name + 1 is the label number for that bone in the labelVolume
        ex. if boneNames[2] = 'Tibia', then the label for tibia in labelVolume is 3.
        '''
        # clear table
        self.ui.nmTable.clearContents()

        numberOfBones = len(boneNames)
        if numberOfBones <= 0:
            return
        # get proper number of rows
        while self.ui.nmTable.rowCount() > numberOfBones:
            self.ui.nmTable.removeRow(self.ui.nmTable.rowCount())
        while self.ui.nmTable.rowCount() < numberOfBones:
            self.ui.nmTable.insertRow(self.ui.nmTable.rowCount())

        columnLabels = ['Region Size (mm^3)', 'Sum (MBq)', 'Average (MBq/mm^3)', 'Minimum (MBq)', '1st Quartile (MBq)', 'Median (MBq)', '3rd Quartile (MBq)', 'Maximum (MBq)', 'Standard Deviation (MBq)', 'Variance (MBq)']
        # set up columns
        while self.ui.nmTable.columnCount() > len(columnLabels):
            self.ui.nmTable.removeColumn(self.ui.nmTable.columnCount())
        while self.ui.nmTable.columnCount() < len(columnLabels):
            self.ui.nmTable.insertColumn(self.ui.nmTable.columnCount())
        self.ui.nmTable.setHorizontalHeaderLabels(columnLabels)
        self.ui.nmTable.setVerticalHeaderLabels(boneNames)

        # set contents of table
        for i in xrange(len(boneNames)):
            uptakeData = nmField[labelVolume == i + 1] / self.scanData.nmDCM[0x6001, 0x10a1].value / 1000.0  # get scan data in MBq
            regionSize = len(uptakeData) * self.scanData.nmDCM.SliceThickness ** 3  # get volume size in cubic mm
            if len(uptakeData) == 0:
                uptakeData = numpy.zeros(1)
            self.ui.nmTable.setItem(i, 0, QtGui.QTableWidgetItem(str(regionSize)))
            self.ui.nmTable.setItem(i, 1, QtGui.QTableWidgetItem(str(numpy.sum(uptakeData))))
            if regionSize == 0:
                self.ui.nmTable.setItem(i, 2, QtGui.QTableWidgetItem('0.0'))
            else:
                self.ui.nmTable.setItem(i, 2, QtGui.QTableWidgetItem(str(numpy.sum(uptakeData) / regionSize)))
            self.ui.nmTable.setItem(i, 3, QtGui.QTableWidgetItem(str(numpy.min(uptakeData))))
            self.ui.nmTable.setItem(i, 4, QtGui.QTableWidgetItem(str(scipy.stats.scoreatpercentile(uptakeData, 25))))
            self.ui.nmTable.setItem(i, 5, QtGui.QTableWidgetItem(str(numpy.median(uptakeData))))
            self.ui.nmTable.setItem(i, 6, QtGui.QTableWidgetItem(str(scipy.stats.scoreatpercentile(uptakeData, 75))))
            self.ui.nmTable.setItem(i, 7, QtGui.QTableWidgetItem(str(numpy.max(uptakeData))))
            self.ui.nmTable.setItem(i, 8, QtGui.QTableWidgetItem(str(numpy.std(uptakeData))))
            self.ui.nmTable.setItem(i, 9, QtGui.QTableWidgetItem(str(numpy.var(uptakeData))))

        # resize table
        self.ui.nmTable.resizeColumnsToContents()

    def loadModel(self, filepath=None, **kwargs):
        # TODO: split this function up
        '''
        set stlShortcutFilepath to a stl path so that loading dicom will skip actually polygonizing volume field
        '''
        if filepath is None:
            filepaths, flter = QtGui.QFileDialog.getOpenFileNames(caption="Select a model to load", filter="All supported Files (*.stl *.binvox *.dcm);;STL Files (*.stl);;Binvox (*.binvox);;DICOM (*.dcm);;Any File (*.*)")  # @UnusedVariable
            for filepath in filepaths:
                self.loadModel(filepath)
            return
        if len(filepath) == 0:
            return
        model = None
        jName = 'Model Base'
        fileName, fileExtension = os.path.splitext(filepath)  # @UnusedVariable
        if fileExtension == '.stl':
            model = readSTLfile(filepath, True, **kwargs)
            jName = 'Model Base'
        elif fileExtension == '.binvox':
            bv = BinVox(filepath, True)
#            model = bv.createVoxelModel()
            mesh = Polygonise(bv.data, 0.5)
            model = mesh.isosurface(**kwargs)
            jName = 'BinVox Base'
        elif fileExtension == '.dcm':
            dcm = dicom.read_file(filepath)
            field = numpy.array(dcm.pixel_array, dtype=numpy.float)
            field = field * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

            if 'sigma' in kwargs:
                sigma = kwargs['sigma']
            else:
                sigma, ok = QtGui.QInputDialog.getInt(self, 'Gaussian Filter Sigma', 'Enter Sigma for Gaussian Filter, 0 means no filter', maxValue=10, minValue=0, value=2)
                if not ok:
                    sigma = 2
            if sigma != 0:
                field = gaussian_filter(field, sigma)

            if 'resampleFactor' in kwargs:
                resampleFactor = kwargs['resampleFactor']
            else:
                resampleFactor, ok = QtGui.QInputDialog.getInt(None, 'Rescale', "Enter Rescale factor. 1 means don't resample, 2 means 1/2 points, 3 means 1/3 points,...", value=3, minValue=1, maxValue=min(field.shape))
                if not ok:
                    resampleFactor = 3
            if resampleFactor > 1:
                from scipy import ndimage
                # resample at half the original size so as to make polygonising much faster
                # TODO: ensure the math on this for values other than 2
                coord = numpy.mgrid[0:len(field):resampleFactor, 0:len(field[0]):resampleFactor, 0:len(field[0, 0]):resampleFactor]
                field = ndimage.map_coordinates(field, coord)
            if 'stlShortcutFilepath' in kwargs and os.path.exists(kwargs['stlShortcutFilepath']):
                model = self.loadModel(kwargs['stlShortcutFilepath'])
                model.referenceVolume = field
            else:
                # TODO ask for isolevel
                mesh = Polygonise(field, 350.0)
                pd = QtGui.QProgressDialog("Polygonising DICOM file", "Cancel", 0, 100)
                pd.setCancelButton(None)
                model = mesh.isosurface(progressDialogCallback=pd.setValue, **kwargs)
                model.invertNormals()
                pd.setValue(100)
            jName = 'DICOM Base'

        if model is not None:
            self.referenceModelJoint = Joint(name=jName, parentJoint=self.glWidget.worldJoint)
            model.setJoint(self.referenceModelJoint)
            self.setupSceneControls()
            self.glWidget.resizeGL(self.glWidget.width(), self.glWidget.height())
            self.glWidget.updateGL()

        return model

    def loadAtlas(self, filepath=None):
        UserAtlas(filepath, self.glWidget.worldJoint)
        # reset view
        self.glWidget.resizeGL(self.glWidget.width(), self.glWidget.height())
        self.glWidget.updateGL()
        self.setView()
        self.setupSceneControls()

    def setModelVisibility(self, state):
        if self.selectedModel is None:
            return
        if state == QtCore.Qt.CheckState.Checked:
            self.selectedModel.visible = True
        else:
            self.selectedModel.visible = False
        self.glWidget.updateGL()

    def setColorModel(self, state):
        self.ui.chooseAmbientColorMat.setEnabled(True)
        self.ui.matShinySlider.setEnabled(True)
        self.ui.matShinySpinBox.setEnabled(True)
        if state == QtCore.Qt.CheckState.Checked:
            self.glWidget.setColorDrivenMaterial(True)
            self.ui.chooseDiffuseColorMat.setEnabled(False)
            self.ui.chooseSpecularColorMat.setEnabled(False)
            self.ui.chooseEmissiveColorMat.setEnabled(False)
        else:
            self.glWidget.setColorDrivenMaterial(False)
            self.ui.chooseDiffuseColorMat.setEnabled(True)
            self.ui.chooseSpecularColorMat.setEnabled(True)
            self.ui.chooseEmissiveColorMat.setEnabled(True)

    def setRotateCoord(self, state):
        self.ui.jointXRotSlider.setValue(0)
        self.ui.jointXRotSpinBox.setValue(0.0)
        self.ui.jointYRotSlider.setValue(0)
        self.ui.jointYRotSpinBox.setValue(0.0)
        self.ui.jointZRotSlider.setValue(0)
        self.ui.jointZRotSpinBox.setValue(0.0)
        if state == QtCore.Qt.CheckState.Checked:
            self.ui.xAxisLabel.setText('Elevation')
            self.ui.yAxisLabel.setText('Azimuth')
            self.ui.zAxisLabel.setText('Spin')
            self.ui.rotationOrder.setCurrentIndex(0)
            self.ui.rotationOrder.setEnabled(False)
            self.ui.jointXRotSlider.setMaximum(90)
            self.ui.jointXRotSlider.setMinimum(-90)
            self.ui.jointXRotSpinBox.setMaximum(90.0)
            self.ui.jointXRotSpinBox.setMinimum(-90.0)
        else:
            self.ui.xAxisLabel.setText('X')
            self.ui.yAxisLabel.setText('Y')
            self.ui.zAxisLabel.setText('Z')
            self.ui.rotationOrder.setCurrentIndex(0)
            self.ui.rotationOrder.setEnabled(True)
            self.ui.jointXRotSlider.setMaximum(360)
            self.ui.jointXRotSlider.setMinimum(0)
            self.ui.jointXRotSpinBox.setMaximum(360.0)
            self.ui.jointXRotSpinBox.setMinimum(0.0)

    def rotationOrderChanged(self, order):
        self.rotateJointSpinBox()

    def rotateTolSlider(self, value=None):
        QtCore.QObject.disconnect(self.ui.tolXRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateTolSpinBox)
        QtCore.QObject.disconnect(self.ui.tolYRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateTolSpinBox)
        QtCore.QObject.disconnect(self.ui.tolZRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateTolSpinBox)
        x = self.ui.tolXRotSlider.value()
        y = self.ui.tolYRotSlider.value()
        z = self.ui.tolZRotSlider.value()
        self.ui.tolXRotSpinBox.setValue(x)
        self.ui.tolYRotSpinBox.setValue(y)
        self.ui.tolZRotSpinBox.setValue(z)
        QtCore.QObject.connect(self.ui.tolXRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateTolSpinBox)
        QtCore.QObject.connect(self.ui.tolYRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateTolSpinBox)
        QtCore.QObject.connect(self.ui.tolZRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateTolSpinBox)

        self.updateTolVec()

    def rotateTolSpinBox(self, value=None):
        QtCore.QObject.disconnect(self.ui.tolXRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateTolSlider)
        QtCore.QObject.disconnect(self.ui.tolYRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateTolSlider)
        QtCore.QObject.disconnect(self.ui.tolZRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateTolSlider)
        x = self.ui.tolXRotSpinBox.value()
        y = self.ui.tolYRotSpinBox.value()
        z = self.ui.tolZRotSpinBox.value()
        self.ui.tolXRotSlider.setValue(x)
        self.ui.tolYRotSlider.setValue(y)
        self.ui.tolZRotSlider.setValue(z)
        QtCore.QObject.connect(self.ui.tolXRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateTolSlider)
        QtCore.QObject.connect(self.ui.tolYRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateTolSlider)
        QtCore.QObject.connect(self.ui.tolZRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateTolSlider)

        self.updateTolVec()

    def updateTolVec(self):
        # TODO: fill in stuff
        rx = math.radians(self.ui.tolXRotSpinBox.value())
        ry = math.radians(self.ui.tolYRotSpinBox.value())
        rz = math.radians(self.ui.tolZRotSpinBox.value())
        l = self.ui.tolVecLength.value()
        x = self.ui.tolVecXOrigin.value()
        y = self.ui.tolVecYOrigin.value()
        z = self.ui.tolVecZOrigin.value()

        if self.tolJoint is None:
            self.makeTolVec()

        R = numpyTransform.rotation(rz, [0, 0, 1], N=4) * numpyTransform.rotation(ry, [0, 1, 0], N=4) * numpyTransform.rotation(rx, [1, 0, 0], N=4)
        vec = l * numpy.array([0.0, 0.0, 1.0])
        vec = numpyTransform.transformPoints(R, vec[numpy.newaxis, :]).squeeze()
        self.ui.tolVecX.setValue(vec[0])
        self.ui.tolVecY.setValue(vec[1])
        self.ui.tolVecZ.setValue(vec[2])

        self.tolJoint.translate([x, y, z], absolute=True)
        self.tolJoint.rotate(R, relative=False)
        self.tolJoint.scale(l)

        # reset view
#        self.glWidget.resizeGL(self.glWidget.width(), self.glWidget.height())
        self.glWidget.updateGL()
#        self.setView()
#        self.setupSceneControls()

    def makeTolVec(self):
        a = math.radians(self.ui.tolVecAngle.value())
        if self.tolJoint is None:
            self.tolJoint = Joint([0.0, 0.0, 0.0], parentJoint=self.glWidget.worldJoint, name='Tolerance Joint')
        model = TriModel.createTolRegion(a, name='Tolerance Region', color=[1, 0, 0])
        if len(self.tolJoint.models) > 3:
            del self.tolJoint.models[-1]
        model.setJoint(self.tolJoint)

        # reset view
        self.glWidget.resizeGL(self.glWidget.width(), self.glWidget.height())
        self.glWidget.updateGL()
        self.setView()
        self.setupSceneControls()

    def rotateJointSlider(self, value=None):
        x = self.ui.jointXRotSlider.value()
        y = self.ui.jointYRotSlider.value()
        z = self.ui.jointZRotSlider.value()
        self.rotateJoint(x, y, z)

    def rotateJointSpinBox(self, value=None):
        x = self.ui.jointXRotSpinBox.value()
        y = self.ui.jointYRotSpinBox.value()
        z = self.ui.jointZRotSpinBox.value()
        self.rotateJoint(x, y, z)

    def rotateJoint(self, x, y, z):
        if self.selectedJoint is not None:
            if self.ui.sphericalCoord.checkState() == QtCore.Qt.CheckState.Checked:
                sc = True
            else:
                sc = False
#            start = time.time()
            if self.glWidget.useCallLists:
                self.selectedJoint.rotate(x, y, z, sphericalCoord=sc, relative=False, unitsDegrees=True, angleOrder=self.ui.rotationOrder.currentText(), updateModels=False)
            else:
                self.selectedJoint.rotate(x, y, z, sphericalCoord=sc, relative=False, unitsDegrees=True, angleOrder=self.ui.rotationOrder.currentText())
#            print 'Time to rotate: %f' % (time.time()-start)
            self.updateJointPropertiesTab()
#            start = time.time()
            self.glWidget.updateGL()
#            print 'Time to update OpenGL: %f' % (time.time()-start)

    def moveJointX(self, deltaX=None, joint=None):
        if deltaX is None:
            deltaX = self.ui.jointXPosStepSize.value()
        if joint is None:
            joint = self.selectedJoint
        if joint is not None:
            joint.translate([deltaX, 0.0, 0.0], False)
            self.glWidget.updateGL()
            self.updateJointPropertiesTab()

    def moveJointY(self, deltaY=None, joint=None):
        if deltaY is None:
            deltaY = self.ui.jointYPosStepSize.value()
        if joint is None:
            joint = self.selectedJoint
        if joint is not None:
            joint.translate([0.0, deltaY, 0.0], False)
            self.glWidget.updateGL()
            self.updateJointPropertiesTab()

    def moveJointZ(self, deltaZ=None, joint=None):
        if deltaZ is None:
            deltaZ = self.ui.jointZPosStepSize.value()
        if joint is None:
            joint = self.selectedJoint
        if joint is not None:
            joint.translate([0.0, 0.0, deltaZ], False)
            self.glWidget.updateGL()
            self.updateJointPropertiesTab()

    def itemChangedCB(self):
        if self.ui.itemTreeList.currentItem() is not None:
            obj = self.ui.itemTreeList.currentItem().obj
            if isinstance(obj, Joint):
                self.selectedJoint = obj
                self.ui.jointXRotSlider.setEnabled(True)
                self.ui.jointYRotSlider.setEnabled(True)
                self.ui.jointZRotSlider.setEnabled(True)
                self.ui.jointXRotSpinBox.setEnabled(True)
                self.ui.jointYRotSpinBox.setEnabled(True)
                self.ui.jointZRotSpinBox.setEnabled(True)
                if self.ui.sphericalCoord.checkState() == QtCore.Qt.CheckState.Unchecked:
                    self.ui.rotationOrder.setEnabled(True)
                self.updateJointPropertiesTab()
            elif isinstance(obj, TriModel.TriModel):
                self.selectedModel = obj
                self.ui.modelVisible.setEnabled(True)
                self.updateModelPropertiesTab()

    def setSlice100Axis(self, index):
        if self.scanData is not None and self.scanData.slice100Joint is not None:  # ensure there is a joint to move
            # disconnect signals before changing them so as not to cause recursion
            QtCore.QObject.disconnect(self.ui.spinBox100AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice100Axis)
            QtCore.QObject.disconnect(self.ui.slider100AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice100Axis)
            self.ui.spinBox100AxisSlice.setValue(index)
            self.ui.slider100AxisSlice.setValue(index)
            QtCore.QObject.connect(self.ui.spinBox100AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice100Axis)
            QtCore.QObject.connect(self.ui.slider100AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice100Axis)
            self.scanData.slice100Joint.translate([index, 0, 0], absolute=True)
            self.scanData.setSlice100Index(index)
            self.glWidget.updateGL()

    def setSlice010Axis(self, index):
        if self.scanData is not None and self.scanData.slice100Joint is not None:  # ensure there is a joint to move
            # disconnect signals before changing them so as not to cause recursion
            QtCore.QObject.disconnect(self.ui.spinBox010AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice010Axis)
            QtCore.QObject.disconnect(self.ui.slider010AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice010Axis)
            self.ui.spinBox010AxisSlice.setValue(index)
            self.ui.slider010AxisSlice.setValue(index)
            QtCore.QObject.connect(self.ui.spinBox010AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice010Axis)
            QtCore.QObject.connect(self.ui.slider010AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice010Axis)
            self.scanData.slice010Joint.translate([0, index, 0], absolute=True)
            self.scanData.setSlice010Index(index)
            self.glWidget.updateGL()

    def setSlice001Axis(self, index):
        if self.scanData is not None and self.scanData.slice100Joint is not None:  # ensure there is a joint to move
            # disconnect signals before changing them so as not to cause recursion
            QtCore.QObject.disconnect(self.ui.spinBox001AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice001Axis)
            QtCore.QObject.disconnect(self.ui.slider001AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice001Axis)
            self.ui.spinBox001AxisSlice.setValue(index)
            self.ui.slider001AxisSlice.setValue(index)
            QtCore.QObject.connect(self.ui.spinBox001AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice001Axis)
            QtCore.QObject.connect(self.ui.slider001AxisSlice, QtCore.SIGNAL("valueChanged(int)"), self.setSlice001Axis)
            self.scanData.slice001Joint.translate([0, 0, index], absolute=True)
            self.scanData.setSlice001Index(index)
            self.glWidget.updateGL()

    def updateJointPropertiesTab(self):
        # prevent callbacks from happening
        QtCore.QObject.disconnect(self.ui.jointScaleX, QtCore.SIGNAL("valueChanged(double)"), self.setJointScale)
        QtCore.QObject.disconnect(self.ui.jointScaleY, QtCore.SIGNAL("valueChanged(double)"), self.setJointScale)
        QtCore.QObject.disconnect(self.ui.jointScaleZ, QtCore.SIGNAL("valueChanged(double)"), self.setJointScale)
        QtCore.QObject.disconnect(self.ui.jointName, QtCore.SIGNAL("editingFinished()"), self.setJointName)
        self.ui.jointName.setText(self.selectedJoint.name)
        self.ui.jointScaleX.setValue(self.selectedJoint.scaleMat[0, 0])
        self.ui.jointScaleY.setValue(self.selectedJoint.scaleMat[1, 1])
        self.ui.jointScaleZ.setValue(self.selectedJoint.scaleMat[2, 2])
        QtCore.QObject.connect(self.ui.jointScaleX, QtCore.SIGNAL("valueChanged(double)"), self.setJointScale)
        QtCore.QObject.connect(self.ui.jointScaleY, QtCore.SIGNAL("valueChanged(double)"), self.setJointScale)
        QtCore.QObject.connect(self.ui.jointScaleZ, QtCore.SIGNAL("valueChanged(double)"), self.setJointScale)
        QtCore.QObject.connect(self.ui.jointName, QtCore.SIGNAL("editingFinished()"), self.setJointName)

        # Be sure that setting the slider value does not cause recursion
        QtCore.QObject.disconnect(self.ui.jointXRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateJointSlider)
        QtCore.QObject.disconnect(self.ui.jointYRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateJointSlider)
        QtCore.QObject.disconnect(self.ui.jointZRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateJointSlider)
        self.ui.jointXRotSlider.setValue(round(math.degrees(self.selectedJoint.xAngle)))
        self.ui.jointYRotSlider.setValue(round(math.degrees(self.selectedJoint.yAngle)))
        self.ui.jointZRotSlider.setValue(round(math.degrees(self.selectedJoint.zAngle)))
        QtCore.QObject.connect(self.ui.jointXRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateJointSlider)
        QtCore.QObject.connect(self.ui.jointYRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateJointSlider)
        QtCore.QObject.connect(self.ui.jointZRotSlider, QtCore.SIGNAL("valueChanged(int)"), self.rotateJointSlider)

        # Be sure that setting the spinbox value does not cause recursion
        QtCore.QObject.disconnect(self.ui.jointXRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateJointSpinBox)
        QtCore.QObject.disconnect(self.ui.jointYRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateJointSpinBox)
        QtCore.QObject.disconnect(self.ui.jointZRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateJointSpinBox)
        self.ui.jointXRotSpinBox.setValue(float(math.degrees(self.selectedJoint.xAngle)))
        self.ui.jointYRotSpinBox.setValue(float(math.degrees(self.selectedJoint.yAngle)))
        self.ui.jointZRotSpinBox.setValue(float(math.degrees(self.selectedJoint.zAngle)))
        QtCore.QObject.connect(self.ui.jointXRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateJointSpinBox)
        QtCore.QObject.connect(self.ui.jointYRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateJointSpinBox)
        QtCore.QObject.connect(self.ui.jointZRotSpinBox, QtCore.SIGNAL("valueChanged(double)"), self.rotateJointSpinBox)

        self.ui.jointXPos.setValue(self.selectedJoint.location[0])
        self.ui.jointYPos.setValue(self.selectedJoint.location[1])
        self.ui.jointZPos.setValue(self.selectedJoint.location[2])
        self.ui.jointQuatW.setValue(self.selectedJoint.orientation.w)
        self.ui.jointQuatX.setValue(self.selectedJoint.orientation.x)
        self.ui.jointQuatY.setValue(self.selectedJoint.orientation.y)
        self.ui.jointQuatZ.setValue(self.selectedJoint.orientation.z)

    def updateModelPropertiesTab(self):
        # prevent callback from happening
        QtCore.QObject.disconnect(self.ui.modelName, QtCore.SIGNAL("editingFinished()"), self.setModelName)
        self.ui.modelName.setText(self.selectedModel.name)
        QtCore.QObject.connect(self.ui.modelName, QtCore.SIGNAL("editingFinished()"), self.setModelName)
        if self.selectedModel.visible:
            self.ui.modelVisible.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.ui.modelVisible.setCheckState(QtCore.Qt.CheckState.Unchecked)
#        self.ui.modelScale.setValue(self.selectedModel.scale)
        r = int(self.selectedModel.ambientColor[0] * 255)
        g = int(self.selectedModel.ambientColor[1] * 255)
        b = int(self.selectedModel.ambientColor[2] * 255)
        a = int(self.selectedModel.ambientColor[3] * 255)
        self.setButtonIconToColor(self.ui.chooseAmbientColorMat, QtGui.QColor.fromRgb(r, g, b, a))
        self.ui.modelAlpha.setValue(a * 255)

        r = int(self.selectedModel.diffuseColor[0] * 255)
        g = int(self.selectedModel.diffuseColor[1] * 255)
        b = int(self.selectedModel.diffuseColor[2] * 255)
        a = int(self.selectedModel.diffuseColor[3] * 255)
        self.setButtonIconToColor(self.ui.chooseDiffuseColorMat, QtGui.QColor.fromRgb(r, g, b, a))

        r = int(self.selectedModel.specularColor[0] * 255)
        g = int(self.selectedModel.specularColor[1] * 255)
        b = int(self.selectedModel.specularColor[2] * 255)
        a = int(self.selectedModel.specularColor[3] * 255)
        self.setButtonIconToColor(self.ui.chooseSpecularColorMat, QtGui.QColor.fromRgb(r, g, b, a))

        r = int(self.selectedModel.emissionColor[0] * 255)
        g = int(self.selectedModel.emissionColor[1] * 255)
        b = int(self.selectedModel.emissionColor[2] * 255)
        a = int(self.selectedModel.emissionColor[3] * 255)
        self.setButtonIconToColor(self.ui.chooseEmissiveColorMat, QtGui.QColor.fromRgb(r, g, b, a))

        # don't need to call callback, shininess is already set
        QtCore.QObject.disconnect(self.ui.matShinySlider, QtCore.SIGNAL("valueChanged(int)"), self.setShininessColorMat)
        QtCore.QObject.disconnect(self.ui.matShinySpinBox, QtCore.SIGNAL("valueChanged(double)"), self.setShininessColorMat)
        self.ui.matShinySlider.setValue(int(round(self.selectedModel.shininess)))
        self.ui.matShinySpinBox.setValue(self.selectedModel.shininess)
        QtCore.QObject.connect(self.ui.matShinySlider, QtCore.SIGNAL("valueChanged(int)"), self.setShininessColorMat)
        QtCore.QObject.connect(self.ui.matShinySpinBox, QtCore.SIGNAL("valueChanged(double)"), self.setShininessColorMat)

    def updateLightPropertiesTab(self, i):
        if self.glWidget.lights[i].enabled:
            self.ui.lightEnable.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.ui.lightEnable.setCheckState(QtCore.Qt.CheckState.Unchecked)
        if self.glWidget.lights[i].directional:
            self.ui.directionalLight.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.ui.directionalLight.setCheckState(QtCore.Qt.CheckState.Unchecked)

        r = int(self.glWidget.lights[i].specularColor[0] * 255)
        g = int(self.glWidget.lights[i].specularColor[1] * 255)
        b = int(self.glWidget.lights[i].specularColor[2] * 255)
        a = int(self.glWidget.lights[i].specularColor[3] * 255)
        self.setButtonIconToColor(self.ui.chooseSpecularColorCam, QtGui.QColor.fromRgb(r, g, b, a))

        r = int(self.glWidget.lights[i].diffuseColor[0] * 255)
        g = int(self.glWidget.lights[i].diffuseColor[1] * 255)
        b = int(self.glWidget.lights[i].diffuseColor[2] * 255)
        a = int(self.glWidget.lights[i].diffuseColor[3] * 255)
        self.setButtonIconToColor(self.ui.chooseDiffuseColorCam, QtGui.QColor.fromRgb(r, g, b, a))

        r = int(self.glWidget.lights[i].ambientColor[0] * 255)
        g = int(self.glWidget.lights[i].ambientColor[1] * 255)
        b = int(self.glWidget.lights[i].ambientColor[2] * 255)
        a = int(self.glWidget.lights[i].ambientColor[3] * 255)
        self.setButtonIconToColor(self.ui.chooseAmbientColorCam, QtGui.QColor.fromRgb(r, g, b, a))

        r = int(self.glWidget.lights[i].emissiveColor[0] * 255)
        g = int(self.glWidget.lights[i].emissiveColor[1] * 255)
        b = int(self.glWidget.lights[i].emissiveColor[2] * 255)
        a = int(self.glWidget.lights[i].emissiveColor[3] * 255)
        self.setButtonIconToColor(self.ui.chooseEmissiveColorCam, QtGui.QColor.fromRgb(r, g, b, a))

        self.ui.lightXPos.setValue(self.glWidget.lights[i].position[0])
        self.ui.lightYPos.setValue(self.glWidget.lights[i].position[1])
        self.ui.lightZPos.setValue(self.glWidget.lights[i].position[2])

    def setView(self, view='ortho1'):
        self.ui.setView.setCurrentIndex(0)
        vMin, vMax = self.glWidget.getBoundingBox()
        self.glWidget.camera.setView(vMin, vMax, view=view)
        self.glWidget.updateGL()

    def setButtonIconToColor(self, button, color):
        size = button.iconSize()
        pixmap = QtGui.QPixmap(size.width(), size.height())
        pixmap.fill(color)
        button.setIcon(QtGui.QIcon(pixmap))

    def setLightXPos(self, pos):
        i = self.ui.lightNum.value()
        self.glWidget.lights[i].position[0] = pos
        self.glWidget.lights[i].updateOpenGl()
        self.glWidget.updateGL()

    def setLightYPos(self, pos):
        i = self.ui.lightNum.value()
        self.glWidget.lights[i].position[1] = pos
        self.glWidget.lights[i].updateOpenGl()
        self.glWidget.updateGL()

    def setLightZPos(self, pos):
        i = self.ui.lightNum.value()
        self.glWidget.lights[i].position[2] = pos
        self.glWidget.lights[i].updateOpenGl()
        self.glWidget.updateGL()

    def setLightEnable(self, state):
        i = self.ui.lightNum.value()
        if state == QtCore.Qt.CheckState.Checked:
            self.glWidget.lights[i].enabled = True
        else:
            self.glWidget.lights[i].enabled = False
        self.glWidget.lights[i].updateOpenGl()
        self.glWidget.updateGL()

    def setLightDirectional(self, state):
        i = self.ui.lightNum.value()
        if state == QtCore.Qt.CheckState.Checked:
            self.glWidget.lights[i].directional = True
            self.glWidget.lights[i].position[3] = 0.0
        else:
            self.glWidget.lights[i].directional = False
            self.glWidget.lights[i].position[3] = 1.0
        self.glWidget.lights[i].updateOpenGl()
        self.glWidget.updateGL()

    def setAmbientColorMat(self):
        if self.selectedModel is None:
            return
#        QtGui.QColorDialog.setOption(QtGui.QColorDialog.ShowAlphaChannel)
        colorDialog = QtGui.QColorDialog()
        colorDialog.setOption(QtGui.QColorDialog.ShowAlphaChannel, True)
        color = colorDialog.getColor()
#        color = QtGui.QColorDialog().getColor(QtGui.QColorDialog.ShowAlphaChannel)
#        color = QtGui.QColorDialog().getColor()

        if color.isValid():
            self.setButtonIconToColor(self.ui.chooseAmbientColorMat, color)
            self.selectedModel.ambientColor = numpy.array(color.getRgb()) / 255.0
            if self.ui.enableColorDrivenModelsCheckBox.checkState() == QtCore.Qt.CheckState.Checked:
                self.setButtonIconToColor(self.ui.chooseDiffuseColorMat, color)
                self.selectedModel.diffuseColor = numpy.array(color.getRgb()) / 255.0
            self.glWidget.updateGL()

    def setDiffuseColorMat(self):
        if self.selectedModel is None:
            return
        color = QtGui.QColorDialog.getColor()
        if color.isValid():
            self.setButtonIconToColor(self.ui.chooseDiffuseColorMat, color)
            self.selectedModel.diffuseColor = numpy.array(color.getRgb()) / 255.0
            self.glWidget.updateGL()

    def setEmissiveColorMat(self):
        if self.selectedModel is None:
            return
        color = QtGui.QColorDialog.getColor()
        if color.isValid():
            self.setButtonIconToColor(self.ui.chooseEmissionColorMat, color)
            self.selectedModel.emissionColor = numpy.array(color.getRgb()) / 255.0
            self.glWidget.updateGL()

    def setSpecularColorMat(self):
        if self.selectedModel is None:
            return
        color = QtGui.QColorDialog.getColor()
        if color.isValid():
            self.setButtonIconToColor(self.ui.chooseSpecularColorMat, color)
            self.selectedModel.specularColor = numpy.array(color.getRgb()) / 255.0
            self.glWidget.updateGL()

    def setShininessColorMat(self, value):
        if self.selectedModel is None:
            return
        # prevent recursion
        QtCore.QObject.disconnect(self.ui.matShinySlider, QtCore.SIGNAL("valueChanged(int)"), self.setShininessColorMat)
        QtCore.QObject.disconnect(self.ui.matShinySpinBox, QtCore.SIGNAL("valueChanged(double)"), self.setShininessColorMat)
        self.ui.matShinySlider.setValue(round(value))
        self.ui.matShinySpinBox.setValue(float(value))
        QtCore.QObject.connect(self.ui.matShinySlider, QtCore.SIGNAL("valueChanged(int)"), self.setShininessColorMat)
        QtCore.QObject.connect(self.ui.matShinySpinBox, QtCore.SIGNAL("valueChanged(double)"), self.setShininessColorMat)

        self.selectedModel.shininess = float(value)
        self.glWidget.updateGL()

    def setSpecularColorCam(self):
        color = QtGui.QColorDialog.getColor()
        if color.isValid():
            self.setButtonIconToColor(self.ui.chooseSpecularColorCam, color)
            i = self.ui.lightNum.value()
            self.glWidget.lights[i].specularColor = numpy.array(color.getRgb()) / 255.0
            self.glWidget.lights[i].updateOpenGl()
            self.glWidget.updateGL()

    def setDiffuseColorCam(self):
        color = QtGui.QColorDialog.getColor()
        if color.isValid():
            self.setButtonIconToColor(self.ui.chooseDiffuseColorCam, color)
            i = self.ui.lightNum.value()
            self.glWidget.lights[i].diffuseColor = numpy.array(color.getRgb()) / 255.0
            self.glWidget.lights[i].updateOpenGl()
            self.glWidget.updateGL()

    def setAmbientColorCam(self):
        color = QtGui.QColorDialog.getColor()
        if color.isValid():
            self.setButtonIconToColor(self.ui.chooseAmbientColorCam, color)
            i = self.ui.lightNum.value()
            self.glWidget.lights[i].ambientColor = numpy.array(color.getRgb()) / 255.0
            self.glWidget.lights[i].updateOpenGl()
            self.glWidget.updateGL()

    def setEmissiveColorCam(self):
        color = QtGui.QColorDialog.getColor()
        if color.isValid():
            self.setButtonIconToColor(self.ui.chooseEmissiveColorCam, color)
            i = self.ui.lightNum.value()
            self.glWidget.lights[i].emissiveColor = numpy.array(color.getRgb()) / 255.0
            self.glWidget.lights[i].updateOpenGl()
            self.glWidget.updateGL()

    def populateItemTree(self, joint):
        if joint is not None:
            self.ui.itemTreeList.clear()
            self._recursPopulateItemTree(joint)

    def _recursPopulateItemTree(self, joint):
        leaf = modelTreeItem(self.ui.itemTreeList, joint)
        for model in joint.models:
            modelTreeItem(leaf, model)
        for cJoint in joint.childJoints:
            self._recursPopulateItemTree(cJoint)

    def createDefaultScene(self):
        # remove current scene
        self.glWidget.clearScene()

        # create and connect joints
        joint1_2 = Joint([0.5, 0.5, 3.5], parentJoint=self.glWidget.worldJoint, showAxis=True, axisScale=0.5, name='Joint 1-2')
        joint2_3 = Joint([0.5, 0.5, 7.5], parentJoint=joint1_2, showAxis=True, axisScale=0.5, name='Joint 2-3')

        # create bone models
        TriModel.createRectangularSolid([1, 1, 3], [0, 0, 0], joint=self.glWidget.worldJoint, name='Bone 1', color=[0.0, 170.0 / 255, 1.0])
        TriModel.createRectangularSolid([1, 1, 3], [0, 0, 4], joint=joint1_2, name='Bone 2', color=[1.0, 170.0 / 255, 0.0])
        TriModel.createRectangularSolid([1, 1, 3], [0, 0, 8], joint=joint2_3, name='Bone 3', color=[170.0 / 255, 0.0, 1.0])

        # reset view
        self.glWidget.resizeGL(self.glWidget.width(), self.glWidget.height())
        self.setView()
        self.setupSceneControls()

    def clearScene(self):
        self.glWidget.clearScene()
        self.setupSceneControls()


class modelTreeItem(QtGui.QTreeWidgetItem):
    def __init__(self, parent=None, obj=None):
        QtGui.QTreeWidgetItem.__init__(self, parent)
        self.obj = obj
        self.setText(0, self.obj.name)


def main():
    app = QtGui.QApplication(sys.argv)
    myapp = MyMainWindow()
    myapp.show()
    sys.exit(app.exec_())


def mainProfile():
    import cProfile
    '''
    run
        python -m cProfile -o pyAtlasSegmentation.TimeProfile pyAtlasSegmentation.py
    then run
        python runsnake.py pyAtlasSegmentation.TimeProfile
    '''
    if os.path.exists('pyAtlasSegmentation.TimeProfile'):
        os.remove('pyAtlasSegmentation.TimeProfile')
    cProfile.runctx('main()', globals(), locals(), filename='pyAtlasSegmentation.TimeProfile')
    os.system('runsnake pyAtlasSegmentation.TimeProfile')

if __name__ == '__main__':
    main()
#    mainProfile()
