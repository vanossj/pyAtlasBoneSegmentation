'''
Created on Jan 29, 2012

@author: Jeff
'''
import os.path
import numpy, scipy
from scipy.ndimage import gaussian_filter, map_coordinates
from PySide import QtGui, QtCore
import dicom
import TriModel
from Joint import Joint
from polygonise import Polygonise
from STL import readSTLfile, saveSTLFile
from OpenGL import GL
import time
from lxml import etree
from scipy.ndimage.measurements import label
from scipy.spatial import cKDTree as KDTree


class UserAtlas():
    '''
    classdocs
    '''
    def __init__(self, filepath=None, joint=None, **kwargs):
        '''
        filepath
            filepath to xml file that describes the atlas

        joint
            joint

        Key Words:
        joint
        '''
        if filepath is None or not isinstance(filepath, str) or not os.path.exists(filepath):
            filepath, flter = QtGui.QFileDialog.getOpenFileName(caption="Select an Atlas xml file to load", filter="XML (*.xml);;Any File (*.*)")  # @UnusedVariable
        if len(filepath) == 0:
            return

        if joint is None or not isinstance(joint, Joint):
            return

        # set up variables
        self.atlasAxes = numpy.identity(3)
        self.atlasModels = {}

        # Start loading atlas
        xmldoc = etree.parse(filepath)
        root = xmldoc.getroot()
        numBones = 0
        for axesElement in root.iter(tag=etree.Element):
            if axesElement.tag == 'axes':
                for vectorElement in axesElement:
                    if vectorElement.tag == 'anteriorVector':
                        v = vectorElement.text[1:-1].split(',')
                        self.atlasAxes[0] = [float(v[0]), float(v[1]), float(v[2])]
                    elif vectorElement.tag == 'dorsalVector':
                        v = vectorElement.text[1:-1].split(',')
                        self.atlasAxes[1] = [float(v[0]), float(v[1]), float(v[2])]
                    elif vectorElement.tag == 'rightVector':
                        v = vectorElement.text[1:-1].split(',')
                        self.atlasAxes[2] = [float(v[0]), float(v[1]), float(v[2])]
                break  # should only be one 'axes' element

        for element in root.iter('bone'):  # @UnusedVariable
            numBones += 1
        progress = QtGui.QProgressDialog("Loading Atlas...", "Abort", 0, numBones, None)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        atlasBasePosition = [0.0, 0.0, 0.0]
        for posElement in root.iter(tag=etree.Element):
            if posElement.tag == 'position':
                p = posElement.text[1:-1].split(',')
                atlasBasePosition = [float(p[0]), float(p[1]), float(p[2])]
                break

        self.atlasJoint = Joint(atlasBasePosition, name='Atlas Base', parentJoint=joint)
        self._loadAtlasJointInfo(self.atlasJoint, root)
        self._loadAtlas(root, self.atlasJoint, filepath, progress)
        progress.setValue(numBones)

    def _loadAtlasJointInfo(self, joint, jointElement):
        for e in jointElement.iter(tag=etree.Element):
            if e.tag == 'jointType':
                joint.type = e.text.strip()
                break
        for e in jointElement.iter(tag=etree.Element):
            if e.tag == 'proximodistalVec':
                v = e.text.strip()[1:-1].split(',')
                joint.proximodistalVec = numpy.array([float(v[0]), float(v[1]), float(v[2])])
                joint.proximodistalVec /= numpy.linalg.norm(joint.proximodistalVec)  # normalize
                joint.proximodistalVecTransformed = joint.proximodistalVec.copy()
                break
        for e in jointElement.iter(tag=etree.Element):
            if e.tag == 'secondaryVec':
                v = e.text.strip()[1:-1].split(',')
                joint.secondaryVec = numpy.array([float(v[0]), float(v[1]), float(v[2])])
                joint.secondaryVec /= numpy.linalg.norm(joint.secondaryVec)  # normalize
                joint.secondaryVecTransformed = joint.secondaryVec.copy()
                break
        for e in jointElement.iter(tag=etree.Element):
            if e.tag == 'tertiaryVec':
                v = e.text.strip()[1:-1].split(',')
                joint.tertiaryVec = numpy.array([float(v[0]), float(v[1]), float(v[2])])
                joint.tertiaryVec /= numpy.linalg.norm(joint.tertiaryVec)  # normalize
                joint.tertiaryVecTransformed = joint.tertiaryVec.copy()
                break
        for e in jointElement.iter(tag=etree.Element):
            if e.tag == 'length':
                joint.length = float(e.text.strip())
                break
        for e in jointElement.iter(tag=etree.Element):
            if e.tag == 'DOFvec':
                v = e.text.strip()[1:-1].split(',')
                joint.DOFvec = numpy.array([float(v[0]), float(v[1]), float(v[2])])
                break
        for e in jointElement.iter(tag=etree.Element):
            if e.tag == 'DOFangle':
                joint.DOFangle = numpy.radians(float(e.text.strip()))
                break
        for e in jointElement.iter(tag=etree.Element):
            if e.tag == 'DOFtrans':
                joint.DOFtrans = float(e.text.strip())
                break

        joint.proximodistalVecScaled = joint.length * joint.proximodistalVec

    def _loadAtlas(self, parentElement, parentJoint, origfilepath=None, progressDialog=None):
        for element in parentElement:
            if progressDialog is not None and progressDialog.wasCanceled():
                break
            # attach bones to joint
            if element.tag == 'bone':
                # find bone model filepath
                filepath = None
                for pathElement in element:
                    if pathElement.tag == 'filepath':
                        filepath = pathElement.text
                        break
                if os.path.exists(filepath) is False and origfilepath is not None:
                        # convert relative filepath (from xml location) to absolute filepaths
                    head, tail = os.path.split(origfilepath)  # @UnusedVariable
                    filepath = os.path.join(head, filepath)
                if os.path.exists(filepath) is True:
                    for colorElement in element:
                        color = None
                        if colorElement.tag == 'color':
                            c = colorElement.text[1:-1].split(',')
                            if len(c) >= 3:
                                color = [float(c[0]) / 255.0, float(c[1]) / 255.0, float(c[2]) / 255.0]
                                if len(c) >= 4:
                                    color.append(float(c[3]) / 255.0)
                    visible = True
                    for visibleElement in element:
                        if visibleElement.tag == 'visible':
                            if visibleElement.text == 'False':
                                visible = False
                    model = readSTLfile(filepath, True, color=color, visible=visible)
                    model.name = element.text.strip()
                    model.setJoint(parentJoint)
                    self.atlasModels[model.name] = model
                    print 'Model ID %d is %s' % (id(model), model.name)

                # update progress dialog
                if progressDialog is not None:
                    progressDialog.setValue(progressDialog.value() + 1)

            if element.tag == 'joint':
                pos = None
                # search for position
                for posElement in element:
                    if posElement.tag == 'position':
                        p = posElement.text[1:-1].split(',')
                        pos = [float(p[0]), float(p[1]), float(p[2])]
                        break
                if pos is None:
                    continue
                # create joint
                joint = Joint(pos, parentJoint=parentJoint, showAxis=True, axisScale=0.5, name=element.text.strip())
                self._loadAtlasJointInfo(joint, element)
                # recursively call load atlas to add children joints
                self._loadAtlas(element, joint, origfilepath, progressDialog)

    def resetAtlas(self):
        # TODO: create resetAtlas function
        pass

    def createSTL(self, filepath=''):
#        transformedVertexList =numpy.array([])
#        TriangleVertexIndexList = numpy.array([])
#        NormVectors = numpy.array([])
        self._createSTL(self.atlasJoint)
#        transformedVertexList, TriangleVertexIndexList, NormVectors = self._getModelData(self.atlasJoint, transformedVertexList, TriangleVertexIndexList, NormVectors)
#        #TODO: make trimodel from data
#        model = TriModel.TriModel(transformedVertexList, TriangleVertexIndexList, joint=self.atlasJoint, normalVectors=NormVectors)
#        saveSTLFile(model, filepath, binary=False)

    def _createSTL(self, joint):
        for model in joint.models:
            if model.visible and model.name[:5] != 'axis_':  # ignore models that are not visible and models that illustrate the axis
                newModel = TriModel.TriModel(model.transformedVertexList.copy(), model.TriangleVertexIndexList.copy(), normalVectors=model.NormVectors.copy())
                filepath = 'Atlas Bone %d %s.stl' % (id(model), model.name)
                saveSTLFile(newModel, filepath, binary=False)
                print 'Saved STL file ', filepath
        for childJoint in joint.childJoints:
            self._createSTL(childJoint)

    def _getModelData(self, joint, transformedVertexList, TriangleVertexIndexList, NormVectors):
        for model in joint.models:
            if model.visible and model.name[:5] != 'axis_':  # ignore models that are not visible and models that illustrate the axis
                # get new transformedVertexList
                n1 = transformedVertexList.shape[0]
                n2 = model.transformedVertexList.shape[0]
                transformedVertexList = numpy.append(transformedVertexList, model.transformedVertexList)
                transformedVertexList.shape = (n1 + n2, 3)
                # get new VertexList
                n1 = TriangleVertexIndexList.shape[0]
                n2 = model.TriangleVertexIndexList.shape[0]
                TriangleVertexIndexList = numpy.append(TriangleVertexIndexList, model.TriangleVertexIndexList)
                TriangleVertexIndexList.shape = (n1 + n2, 3)
                # get new NormVectors
                n1 = NormVectors.shape[0]
                n2 = model.NormVectors.shape[0]
                NormVectors = numpy.append(NormVectors, model.NormVectors)
                NormVectors.shape = (n1 + n2, 3)
        for childJoint in joint.childJoints:
            transformedVertexListJ, TriangleVertexIndexListJ, NormVectorsJ = self._getModelData(childJoint, transformedVertexList, TriangleVertexIndexList, NormVectors)
            # get new transformedVertexList
            n1 = transformedVertexList.shape[0]
            n2 = transformedVertexListJ.shape[0]
            transformedVertexList = numpy.append(transformedVertexList, transformedVertexListJ)
            transformedVertexList.shape = (n1 + n2, 3)
            # get new VertexList
            n1 = TriangleVertexIndexList.shape[0]
            n2 = TriangleVertexIndexListJ.shape[0]
            TriangleVertexIndexList = numpy.append(TriangleVertexIndexList, TriangleVertexIndexListJ)
            TriangleVertexIndexList.shape = (n1 + n2, 3)
            # get new NormVectors
            n1 = NormVectors.shape[0]
            n2 = NormVectorsJ.shape[0]
            NormVectors = numpy.append(NormVectors, NormVectorsJ)
            NormVectors.shape = (n1 + n2, 3)

        return transformedVertexList, TriangleVertexIndexList, NormVectors


class UserDicom():
    '''
    classdocs
    '''

    def __init__(self, ct=None, nm=None, **kwargs):
        '''
        ct
            filepath or dicom object of uCT data

        nm
            filepath or dicom object of uSPECT or uPET data, this data is fused with uCT data

        Key Words:
        joint
        sigma
        resampleFactor
        isolevel
        '''

        if ct is None:
            ct, flter = QtGui.QFileDialog.getOpenFileName(caption="Select a uCT dicom file to load", filter="DICOM (*.dcm);;Any File (*.*)")  # @UnusedVariable
            if len(ct) == 0:
                return
        if nm is None:
            nm, flter = QtGui.QFileDialog.getOpenFileName(caption="Select a uSPECT or uPET dicom file to load", filter="DICOM (*.dcm);;Any File (*.*)")  # @UnusedVariable
            if len(nm) == 0:
                return
        if isinstance(ct, (str, unicode)) and os.path.exists(ct):
            ct = dicom.read_file(ct)
        if isinstance(nm, (str, unicode)) and os.path.exists(nm):
            nm = dicom.read_file(nm)
        if not isinstance(ct, dicom.dataset.FileDataset):
            Exception('CT dataset Incorrect data type')
        if not isinstance(nm, dicom.dataset.FileDataset):
            Exception('Nuclear Medicine dataset Incorrect data type')

        # set up variables
        self.ctDCM = ct
        self.nmDCM = nm
        self.slice100Model = None
        self.slice010Model = None
        self.slice001Model = None
        self.CTaxis100TextureIDs = None
        self.CTaxis010TextureIDs = None
        self.CTaxis001TextureIDs = None
        self.isosurfaceModel = None
        self.currentAxis100TextureIDs = None
        self.currentAxis010TextureIDs = None
        self.currentAxis001TextureIDs = None
        self.CTaxis100TextureIDs = None
        self.CTaxis010TextureIDs = None
        self.CTaxis001TextureIDs = None
        self.LabeledVolumeaxis100TextureIDs = None
        self.LabeledVolumeaxis010TextureIDs = None
        self.LabeledVolumeaxis001TextureIDs = None
        self.alignmentAxis = None

        if 'joint' in kwargs and isinstance(kwargs['joint'], Joint):
            self.joint = kwargs['joint']
            self.isosurfaceJoint = Joint(parentJoint=self.joint, name='Isosurface')
            self.slice100Joint = Joint(parentJoint=self.joint, name='Slice 100 Joint')
            self.slice010Joint = Joint(parentJoint=self.joint, name='Slice 010 Joint')
            self.slice001Joint = Joint(parentJoint=self.joint, name='Slice 001 Joint')
        else:
            self.joint = None

        self.isolevel = 350.0
        if 'isolevel' in kwargs:
            self.isolevel = kwargs['isolevel']
        else:
            dialogIsolevel, ok = QtGui.QInputDialog.getDouble(None, 'Choose Isolevel', 'Enter the Hounsfield value to use as the isolevel to create the skeleton isosurface from the CT data', maxValue=1000.0, minValue=-1000.0, value=self.isolevel)
            if ok:
                self.isolevel = dialogIsolevel

        # Get uCT field in hounsfield units
        self.ctField = numpy.array(self.ctDCM.pixel_array, dtype=numpy.float)
        self.ctField = self.ctField * float(self.ctDCM.RescaleSlope) + float(self.ctDCM.RescaleIntercept)
        # TODO: read in pixel res
        try:
            self.sliceThickness = float(self.ctDCM.SliceThickness)
        except:
            raise Exception("Can't find slice thickness in CT dicom")

        # get nm field
        self.nmField = numpy.array(self.nmDCM.pixel_array, dtype=numpy.float)

        # Create field to do computations on
        self.ctFieldPreprocessed = self.ctField.copy()

        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        else:
            self.sigma, ok = QtGui.QInputDialog.getInt(None, 'Gaussian Filter Sigma', 'Enter Sigma for Gaussian Filter, 0 means no filter', maxValue=10, minValue=0, value=2)
            if not ok:
                self.sigma = 0
        if self.sigma != 0:
            self.ctFieldPreprocessed = gaussian_filter(self.ctFieldPreprocessed, self.sigma)

        if 'resampleFactor' in kwargs:
            self.resampleFactor = kwargs['resampleFactor']
        else:
            self.resampleFactor, ok = QtGui.QInputDialog.getInt(None, 'Resample', "Enter Resample factor. 1 means don't resample, 2 means 1/2 points, 3 means 1/3 points,...", value=3, minValue=1, maxValue=min(self.ctField.shape) / 10)
            if not ok:
                self.resampleFactor = 1
        if self.resampleFactor > 1:
            # resample at a fraction of the original size to make future processing much faster
#            coord = numpy.mgrid[0:len(self.ctFieldPreprocessed):self.resampleFactor,0:len(self.ctFieldPreprocessed[0]):self.resampleFactor,0:len(self.ctFieldPreprocessed[0,0]):self.resampleFactor]
            coord = numpy.mgrid[0:self.ctFieldPreprocessed.shape[0]:self.resampleFactor,
                                0:self.ctFieldPreprocessed.shape[1]:self.resampleFactor,
                                0:self.ctFieldPreprocessed.shape[2]:self.resampleFactor]
            self.ctFieldPreprocessed = map_coordinates(self.ctFieldPreprocessed, coord)

#        #setup slice view
#        if self.joint is not None:
#            t1 = time.time()
#            self.CTaxis100TextureIDs, self.CTaxis010TextureIDs, self.CTaxis001TextureIDs = self.createTextures(self.ctField)
#            self.setTextureToCT()
#            self.createSlicesView()
#            print 'Took %f seconds to create slice models & textures' % (time.time()-t1)

        # set up meshs that represent forelimbs and the rest of the body
        # Figure out labels
        labels = []
        ctFieldIso = numpy.zeros(self.ctFieldPreprocessed.shape, dtype=numpy.int)
        ctFieldIso[self.ctFieldPreprocessed >= self.isolevel] = 1
        volLabeled, numLabels = label(ctFieldIso)
        for i in xrange(numLabels):
            size = numpy.sum(volLabeled == i)
            labels.append((i, size))
#            print 'Label %d Size %d' % (i, size)
        labels = numpy.array(labels, dtype=[('label', int), ('size', int)])
        labels = numpy.sort(labels, order='size')
#        volDisp = numpy.zeros(volLabeled.shape)
#        print labels
#        print 'Largest Labels', labels[-2][0], labels[-3][0], labels[-4][0]
#        volDisp[volLabeled == (labels[-2][0])]=1.0
#        volDisp[volLabeled == (labels[-3][0])]=2.0
#        volDisp[volLabeled == (labels[-4][0])]=3.0

        self.ctVolume3 = self.ctFieldPreprocessed.copy()
        self.ctVolume3[numpy.logical_not(volLabeled == (labels[-4][0]))] = 0.0
        self.ctVolume2 = self.ctFieldPreprocessed.copy()
        self.ctVolume2[numpy.logical_not(volLabeled == (labels[-3][0]))] = 0.0
        self.ctVolume1 = self.ctFieldPreprocessed.copy()
        self.ctVolume1[numpy.logical_not(volLabeled == (labels[-2][0]))] = 0.0

        if 'subsection3Shortcut' in kwargs:
            self.isosurfaceModel3 = readSTLfile(kwargs['subsection3Shortcut'], joint=self.isosurfaceJoint)
            self.isosurfaceModel3.name = 'Isosurface 3'
        else:
            pd = QtGui.QProgressDialog("Polygonising SubSection file", "Cancel", 0, 100)
            pd.setCancelButton(None)
            self.isosurfaceModel3 = Polygonise(self.ctVolume3, 350.0).isosurface(progressDialogCallback=pd.setValue, joint=self.isosurfaceJoint)
            self.isosurfaceModel3.invertNormals()
            self.isosurfaceModel3.name = 'Isosurface 3'
            pd.setValue(100)
#            saveSTLFile(self.isosurfaceModel3, 'Isosurface3.stl', binary=False)

        if 'subsection2Shortcut' in kwargs:
            self.isosurfaceModel2 = readSTLfile(kwargs['subsection2Shortcut'], joint=self.isosurfaceJoint)
            self.isosurfaceModel2.name = 'Isosurface 2'
        else:
            pd = QtGui.QProgressDialog("Polygonising SubSection file", "Cancel", 0, 100)
            pd.setCancelButton(None)
            self.isosurfaceModel2 = Polygonise(self.ctVolume2, 350.0).isosurface(progressDialogCallback=pd.setValue, joint=self.isosurfaceJoint)
            self.isosurfaceModel2.invertNormals()
            self.isosurfaceModel2.name = 'Isosurface 2'
            pd.setValue(100)
#            saveSTLFile(self.isosurfaceModel2, 'Isosurface2.stl', binary=False)

        if 'subsection1Shortcut' in kwargs:
            self.isosurfaceModel1 = readSTLfile(kwargs['subsection1Shortcut'], joint=self.isosurfaceJoint)
            self.isosurfaceModel1.name = 'Isosurface 1'
        else:
            pd = QtGui.QProgressDialog("Polygonising SubSection file", "Cancel", 0, 100)
            pd.setCancelButton(None)
            self.isosurfaceModel1 = Polygonise(self.ctVolume1, 350.0).isosurface(progressDialogCallback=pd.setValue, joint=self.isosurfaceJoint)
            self.isosurfaceModel1.invertNormals()
            self.isosurfaceModel1.name = 'Isosurface 1'
            pd.setValue(100)
#            saveSTLFile(self.isosurfaceModel1, 'Isosurface1.stl', binary=False)

        # setup isosurface
        if self.joint is not None:
            if 'isosurfaceModelShortcut' in kwargs:
                self.createIsosurfaceView(stlShortcutFilepath=kwargs['isosurfaceModelShortcut'])
            else:
                self.createIsosurfaceView()
#                v = []
#                v = numpy.append(v, self.isosurfaceModel1.OriginalVertexList.flatten())
#                v = numpy.append(v, self.isosurfaceModel2.OriginalVertexList.flatten())
#                v = numpy.append(v, self.isosurfaceModel3.OriginalVertexList.flatten())
#                v = v.reshape((-1,3))
#                tri = numpy.arange(v.shape[0])
#                tri = tri.reshape((-1,3))
#                self.isosurfaceModel = TriModel.TriModel(v, tri, name='Whole Body IsoSurface', joint=self.isosurfaceJoint)

        self.createCTMask()

        self.bonesVol1 = ['Skull Outside', 'Skull Inside', 'Pelvis Right', 'Upper Hindlimb Right', 'Lower Hindlimb Right', 'HindPaw Right', 'Pelvis Left', 'Upper Hindlimb Left', 'Lower Hindlimb Left', 'HindPaw Left']

    def setTextureToCT(self):
            self.currentAxis100TextureIDs = self.CTaxis100TextureIDs
            self.currentAxis010TextureIDs = self.CTaxis010TextureIDs
            self.currentAxis001TextureIDs = self.CTaxis001TextureIDs

    def setTextureToLabeledVolume(self):
            self.currentAxis100TextureIDs = self.LabeledVolumeaxis100TextureIDs
            self.currentAxis010TextureIDs = self.LabeledVolumeaxis010TextureIDs
            self.currentAxis001TextureIDs = self.LabeledVolumeaxis001TextureIDs

    def createLabelVolumeTextures(self, labeledVolume):
        self.LabeledVolumeaxis100TextureIDs, self.LabeledVolumeaxis010TextureIDs, self.LabeledVolumeaxis001TextureIDs = self.createTextures(labeledVolume)

    def createSlicesView(self):
        # choose initial index
        self.sliceIndex = numpy.array([100, 90, 90], dtype=numpy.int)

        # setup axis [1,0,0]
        vertexList = [[0, 0, 0],
                      [0, self.ctField.shape[1], 0],
                      [0, self.ctField.shape[1], self.ctField.shape[2]],
                      [0, 0, self.ctField.shape[2]]]
        sqrList = [[0, 1, 2, 3]]
        self.slice100Model = TriModel.TriModel(numpy.array(vertexList), numpy.array(sqrList), self.slice100Joint, name='Slice [1,0,0]', textureID=self.CTaxis100TextureIDs[self.sliceIndex[0]])

        # setup axis [0,1,0]
        vertexList = [[0, 0, 0],
                      [self.ctField.shape[0], 0, 0],
                      [self.ctField.shape[0], 0, self.ctField.shape[2]],
                      [0, 0, self.ctField.shape[2]]]
        sqrList = [[0, 1, 2, 3]]
        self.slice010Model = TriModel.TriModel(numpy.array(vertexList), numpy.array(sqrList), self.slice010Joint, name='Slice [0,1,0]', textureID=self.CTaxis010TextureIDs[self.sliceIndex[1]])

        # setup axis [0,0,1]
        vertexList = [[0, 0, 0],
                      [self.ctField.shape[0], 0, 0],
                      [self.ctField.shape[0], self.ctField.shape[1], 0],
                      [0, self.ctField.shape[1], 0]]
        sqrList = [[0, 1, 2, 3]]
        self.slice001Model = TriModel.TriModel(numpy.array(vertexList), numpy.array(sqrList), self.slice001Joint, name='Slice [0,0,1]', textureID=self.CTaxis001TextureIDs[self.sliceIndex[2]])

    def setSlice100Index(self, index):
        if self.slice100Model is not None and index >= 0 and index < self.ctField.shape[0]:  # check for proper index
            self.slice100Model.textureID = self.currentAxis100TextureIDs[index]
            self.slice100Joint.translate([index, 0, 0], absolute=True)

    def setSlice010Index(self, index):
        if self.slice010Model is not None and index >= 0 and index < self.ctField.shape[1]:  # check for proper index
            self.slice010Model.textureID = self.currentAxis010TextureIDs[index]
            self.slice010Joint.translate([0, index, 0], absolute=True)

    def setSlice001Index(self, index):
        if self.slice001Model is not None and index >= 0 and index < self.ctField.shape[2]:  # check for proper index
            self.slice001Model.textureID = self.currentAxis001TextureIDs[index]
            self.slice001Joint.translate([0, 0, index], absolute=True)

    def createTextures(self, volume, colormap=None):
        refVol = numpy.array(volume)
        if refVol.ndim == 4:
            volRGBA = numpy.array(refVol, dtype=numpy.ubyte)
            if refVol.shape[3] == 4:
                pixelFormat = GL.GL_RGBA
            elif refVol.shape[3] == 3:
                pixelFormat = GL.GL_RGB
            else:
                return None, None, None
        elif refVol.ndim == 3:
            # normalize to range 0-1
            refVol -= refVol.min()
            refVol /= refVol.max()
            refVol *= 255
            refVol = numpy.array(refVol, dtype=numpy.ubyte)
            if colormap is None:  # grayscale
                volRGBA = refVol
                pixelFormat = GL.GL_LUMINANCE
            else:  # assign colormap to grayscale values
                colormap = numpy.array(colormap)
                if colormap.shape[1] == 3:
                    pixelFormat = GL.GL_RGB
                elif colormap.shape[1] == 4:
                    pixelFormat = GL.GL_RGBA
                else:
                    return None, None, None
                volRGBA = numpy.empty((refVol.shape[0], refVol.shape[1], refVol.shape[2], colormap.shape[1]), dtype=numpy.ubyte)
                for i in xrange(colormap.shape[0]):
                    volRGBA[refVol == i] = colormap[i]
        else:
            return None, None, None

        # create openGL texture ID array
        axis100TextureIDs = GL.glGenTextures(volRGBA.shape[0])
        for i in xrange(volRGBA.shape[0]):
            image = volRGBA[i, :, :]
            data = image.tostring()
            height = image.shape[0]
            width = image.shape[1]
            GL.glBindTexture(GL.GL_TEXTURE_2D, axis100TextureIDs[i])
            GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, pixelFormat, GL.GL_UNSIGNED_BYTE, data)

        axis010TextureIDs = GL.glGenTextures(volRGBA.shape[1])
        for i in xrange(volRGBA.shape[1]):
            image = volRGBA[:, i, :]
            data = image.tostring()
            height = image.shape[0]
            width = image.shape[1]
            GL.glBindTexture(GL.GL_TEXTURE_2D, axis010TextureIDs[i])
            GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, pixelFormat, GL.GL_UNSIGNED_BYTE, data)

        axis001TextureIDs = GL.glGenTextures(volRGBA.shape[2])
        for i in xrange(volRGBA.shape[2]):
            image = volRGBA[:, :, i]
            data = image.tostring()
            height = image.shape[0]
            width = image.shape[1]
            GL.glBindTexture(GL.GL_TEXTURE_2D, axis001TextureIDs[i])
            GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, pixelFormat, GL.GL_UNSIGNED_BYTE, data)

        return axis100TextureIDs, axis010TextureIDs, axis001TextureIDs

    def createIsosurfaceView(self, stlShortcutFilepath=''):
        if isinstance(stlShortcutFilepath, str) and os.path.exists(stlShortcutFilepath):
            self.isosurfaceModel = readSTLfile(stlShortcutFilepath, True, joint=self.isosurfaceJoint)
        else:
            pd = QtGui.QProgressDialog("Polygonising DICOM file", "Cancel", 0, 100)
            pd.setCancelButton(None)
            self.isosurfaceModel = Polygonise(self.ctFieldPreprocessed, self.isolevel).isosurface(progressDialogCallback=pd.setValue, joint=self.isosurfaceJoint)
            self.isosurfaceModel.invertNormals()
            pd.setValue(100)
        self.isosurfaceModel.name = 'Isosurface from CT'
        self.isosurfaceJoint.scale(self.resampleFactor)

    def createCTMask(self):
        # assume centers of CT and SPECT images are aligned
        ctdim = numpy.array(self.ctVolume1.shape) * self.resampleFactor * float(self.ctDCM.SliceThickness) # ct volume dimension in mm
        spectdim = numpy.array(self.nmField.shape) * float(self.nmDCM.SliceThickness)  # nm dimension in mm

        print(type(ctdim),ctdim)
        print(type(spectdim),spectdim)

        minval = (ctdim - spectdim) / 2.0
        maxval = ctdim - (ctdim - spectdim) / 2.0

        # get slicing coordinates in mm, subtracting spacing() makes it exclusive on the stop value
        coord = numpy.mgrid[minval[0]:maxval[0] - numpy.spacing(maxval[0]):float(self.nmDCM.SliceThickness),
                            minval[1]:maxval[1] - numpy.spacing(maxval[0]):float(self.nmDCM.SliceThickness),
                            minval[2]:maxval[2] - numpy.spacing(maxval[0]):float(self.nmDCM.SliceThickness)]

        # convert mm slicing coordinates to ct slice units
        coord /= self.resampleFactor * float(self.ctDCM.SliceThickness)

        ctVolume1NMSize = map_coordinates(self.ctVolume1, coord)
        print ctVolume1NMSize.shape
        if ctVolume1NMSize.shape != self.nmField.shape:
            raise Exception('created CT mask volume is not the same size as NM volume')
        self.ctVolume1NMmask = numpy.zeros(ctVolume1NMSize.shape, dtype=numpy.int)
        self.ctVolume1NMmask[ctVolume1NMSize >= self.isolevel] = 1
        self.ctVolumeNMmaskOriginLocation = minval
        print 'CT to NM offset is', self.ctVolumeNMmaskOriginLocation

    def createNM_Mask(self, boneVertexLists):
        '''
        boneVertexLists are lists of transformed vertices, each list contains all the vertices from a single bone group
        '''
        dataPoints = []
        boneIndx = []
        for boneVertexList in boneVertexLists:
            dataPoints = numpy.append(dataPoints, boneVertexList)
            if len(boneIndx) == 0:
                boneIndx.append(len(boneVertexList))
            else:
                boneIndx.append(len(boneVertexList) + boneIndx[-1])
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))
        bonesKDTree = KDTree(dataPoints)
        scanBonePoints = numpy.array(numpy.where(self.ctVolume1NMmask == 1)).T
        _d, NN = bonesKDTree.query(scanBonePoints)

        mask = numpy.zeros(self.ctVolume1NMmask.shape, dtype=numpy.int)
        for i in xrange(NN.shape[0]):
            for j in xrange(len(boneIndx)):
                if NN[i] < boneIndx[j]:  # this nearest neighbor is in bone j, mark it as such in the label mask
                    mask[scanBonePoints[i, 0], scanBonePoints[i, 1], scanBonePoints[i, 2]] = j + 1
                    break

        return mask
