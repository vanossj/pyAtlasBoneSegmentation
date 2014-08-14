'''
Created on Jun 30, 2011

@author: grant
'''
import math
import numpy
import time
from OpenGL import GL, GLU
# TODO: try and remove cgkit, use numpy matrix instead
from cgkit.cgtypes import quat
from scipy.spatial import cKDTree as KDTree
import numpyTransform


class TriModel:
    '''A 3D model constructed from triangle faces'''
    def __init__(self, *args, **kwargs):
        '''
        Function Signatures:
        TriModel(vertexList, triangleVertexIndexList)
        TriModel(vertexList, triangleVertexIndexList, joint)

        Arguments {default value}:

        vertexList
            Nx3 array of vertices of the form [[x1,y1,z1], [x2,y2,z2], ...]
        triangleVertexIndexList
            Nx3 array of indices where TriangleVertexIndexList[i] represents the
            ith triangle face of the model in the form
            [vertex1, vertex2, vertex3] where vertex 1, 2, and 3 are indices of
            the vertexList argument
        joint
            Joint object which serves as center of rotation and determines model
            orientation

        Keywords:
        normalVectors
            Nx3 array of indices where normalVectors[i] represents the normal
            vector of the ith triangle face in the form [x, y, z]
        removeDuplicates
            removes duplicate vertices from vertexList. {False}
        name
            name of the model
        visible
            boolean that tells if the model should be drawn or not
        color
            array of floats range 0.0-1.0 of from [R,G,B] or [R,G,B,A].
            If A is not given it is defaults to 1.0. {0.8,0.8,0.8,1.0}
        alpha
            transparency level of model, range 0.0-1.0, transparent to opaque respectively
        updateOnlyFromGrandparentJoints
            If set, orientation changes to parent joint will not cause model
            to move, only orientation to grandparent joints will move model.
            This is useful for models that represent things independent of joint
            orientation, such as axis models. {False}
        '''
        self.OriginalVertexList = numpy.array(args[0])
        self.TriangleVertexIndexList = numpy.array(args[1])
        self.VertexList = self.OriginalVertexList.copy()
        self.transformedVertexList = self.OriginalVertexList.copy()
        self.minPoint = numpy.min(self.OriginalVertexList, axis=0)
        self.maxPoint = numpy.max(self.OriginalVertexList, axis=0)
        self.maxPointTransformed = self.maxPoint.copy()
        self.minPointTransformed = self.minPoint.copy()
        if 'normalVectors' in kwargs and numpy.sum(kwargs['normalVectors']) != 0:  # make sure normal vectors are actually there
            self.NormVectors = numpy.array(kwargs['normalVectors'])
        else:
            self.NormVectors = numpy.zeros((len(self.TriangleVertexIndexList), 3))
            self.calulateNormals()
        self.OriginalNormVectors = self.NormVectors.copy()
        self.joint = None
        self.CoG = None
        self.PCAvectors = None
        self.kdtree = None
        self.initialRotationCenter = numpy.array([0.0, 0.0, 0.0])
        if 'removeDuplicates' in kwargs and isinstance(kwargs['removeDuplicates'], bool) and kwargs['removeDuplicates']:
            self._removeDuplicateVertices()
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'model'
        if 'visible' in kwargs:
            self.visible = kwargs['visible']
        else:
            self.visible = True
        if 'referenceVolume' in kwargs:
            self.referenceVolume = kwargs['referenceVolume']
        else:
            self.referenceVolume = None
        if 'textureID' in kwargs:
            self.textureID = kwargs['textureID']  # must be a properly initialized OpenGL 2D texture
        else:
            self.textureID = None
        if 'displayAsPoints' in kwargs:
            self.displayAsPoints = kwargs['displayAsPoints']
        else:
            self.displayAsPoints = False

        if len(args) > 2:
            try:
                self.setJoint(args[2])
            except Exception:
                pass
        elif 'joint' in kwargs:
            self.setJoint(kwargs['joint'])

        self.ambientColor = [0.8, 0.8, 0.8, 1.0]
        self.diffuseColor = [0.8, 0.8, 0.8, 1.0]
        self.emissionColor = [0.0, 0.0, 0.0, 1.0]
        self.specularColor = [1.0, 1.0, 1.0, 1.0]
        self.shininess = 128.0
        if 'color' in kwargs and kwargs['color'] is not None:
            color = kwargs['color']
            if len(color) == 3:
                color.append(1.0)
                self.ambientColor = color
                self.diffuseColor = color[:]
            elif len(color) == 4:
                self.ambientColor = color
                self.diffuseColor = color[:]
        if 'alpha' in kwargs and kwargs['alpha'] is not None:
            self.ambientColor[3] = kwargs['alpha']
            self.diffuseColor[3] = kwargs['alpha']

        # only update from grandparentjoint, not parent joint
        if 'updateOnlyFromGrandparentJoints' in kwargs and kwargs['updateOnlyFromGrandparentJoints'] is True:
            self.updateOnlyFromGrandparentJoints = True
        else:
            self.updateOnlyFromGrandparentJoints = False

        self.pointSize = 1.0

        # openGL Display List
        self.displayList = None
        if 'skipOpenGL' not in kwargs:
            self.createOpenGLDisplayList()

    def __del__(self):
        if self.displayList is not None:
            GL.glDeleteLists(self.displayList, 1)

    def setScale(self, scale, absolute=True):
        # FIXME: add ability to scale model in addition to joints, this is not applied to transformation function
        if absolute:
            self.scale = scale
        else:
            self.scale *= scale

    def calulateNormals(self):
        for i in xrange(len(self.TriangleVertexIndexList)):
            v0 = self.VertexList[self.TriangleVertexIndexList[i, 0]]
            v1 = self.VertexList[self.TriangleVertexIndexList[i, 1]]
            v2 = self.VertexList[self.TriangleVertexIndexList[i, 2]]
            self.NormVectors[i] = numpy.cross(v2 - v0, v1 - v0)

    def invertNormals(self):
        self.OriginalNormVectors = -1.0 * self.OriginalNormVectors
        self.NormVectors = self.OriginalNormVectors.copy()
        self.createOpenGLDisplayList()

    def setJoint(self, joint):
        self.initialRotationCenter = joint.location.copy()
        self.joint = joint
        self.joint.models.append(self)

    def transformVertices(self, transform=numpy.matrix(numpy.identity(4)), modelID=None):
        if modelID is not None and modelID != id(self):
            return
        self.transformedVertexList = numpyTransform.transformPoints(transform, self.OriginalVertexList)
        self.maxPointTransformed = numpyTransform.transformPoints(transform, self.maxPoint[numpy.newaxis, :])
        self.minPointTransformed = numpyTransform.transformPoints(transform, self.minPoint[numpy.newaxis, :])

        # TODO: not sure that this is necessary
        self.kdtree = KDTree(self.transformedVertexList)

    def getBoundingBox(self, transform=numpy.matrix(numpy.identity(4))):
        if self.minPoint is None or self.maxPoint is None:
            return None, None
        minP = numpy.matrix([self.minPoint[0], self.minPoint[1], self.minPoint[2], 1.0])
        maxP = numpy.matrix([self.maxPoint[0], self.maxPoint[1], self.maxPoint[2], 1.0])
        minP = transform * minP.T
        maxP = transform * maxP.T
        minP = numpy.array(minP).squeeze()[:3]
        maxP = numpy.array(maxP).squeeze()[:3]
        return minP, maxP

    # TODO: Try using VBO to render dynamic objects, and displaylists for static objects
    def createOpenGLDisplayList(self):
        if self.displayList is None:
            self.displayList = GL.glGenLists(1)
            error = GL.glGetError()
            if error != GL.GL_NO_ERROR:
                print "An OpenGL error has occurred: ", GLU.gluErrorString(error)
                return
        GL.glNewList(self.displayList, GL.GL_COMPILE)
        if self.displayAsPoints:
            GL.glBegin(GL.GL_POINTS)
        else:
            GL.glBegin(GL.GL_TRIANGLES)
        for i in xrange(len(self.TriangleVertexIndexList)):
            if len(self.TriangleVertexIndexList[i]) != 3:
                continue
            GL.glNormal3fv(self.NormVectors[i])
            GL.glVertex3fv(self.VertexList[self.TriangleVertexIndexList[i, 0]])
            GL.glVertex3fv(self.VertexList[self.TriangleVertexIndexList[i, 1]])
            GL.glVertex3fv(self.VertexList[self.TriangleVertexIndexList[i, 2]])
        GL.glEnd()
        if self.displayAsPoints:
            GL.glBegin(GL.GL_POINTS)
        else:
            GL.glBegin(GL.GL_QUADS)
        for i in xrange(len(self.TriangleVertexIndexList)):
            if len(self.TriangleVertexIndexList[i]) != 4:
                continue

            GL.glNormal3fv(self.NormVectors[i])
            if self.textureID is not None:
                GL.glTexCoord2d(0.0, 0.0)
            GL.glVertex3fv(self.VertexList[self.TriangleVertexIndexList[i, 0]])
            if self.textureID is not None:
                GL.glTexCoord2d(0.0, 1.0)
            GL.glVertex3fv(self.VertexList[self.TriangleVertexIndexList[i, 1]])
            if self.textureID is not None:
                GL.glTexCoord2d(1.0, 1.0)
            GL.glVertex3fv(self.VertexList[self.TriangleVertexIndexList[i, 2]])
            if self.textureID is not None:
                GL.glTexCoord2d(1.0, 0.0)
            GL.glVertex3fv(self.VertexList[self.TriangleVertexIndexList[i, 3]])
        GL.glEnd()

        GL.glEndList()

    # Display initial model, but modify its position/orientation with openGL matrix transformations
    def OpenGLPaint(self, colorDrivenMaterial=None, useCallLists=True):
        if self.visible:
            # TODO: add support for updateOnlyFrom GrandparentJoints flag
#             localTransform = grandParentTransformMat
#             #set orientation
#             if not self.updateOnlyFromGrandparentJoints:
#                 localTransform *= self.joint.
#                 q = self.joint.orientation

            # set point size if required
            if self.displayAsPoints:
                GL.glPointSize(self.pointSize)

            # set up model material
            if colorDrivenMaterial is not None:
                GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, self.shininess)
                if colorDrivenMaterial:
                    GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
                    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_EMISSION, [0.0, 0.0, 0.0, 1.0])
                    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
                    GL.glColor4fv(self.ambientColor)
                else:
                    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT, self.ambientColor)
                    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE, self.diffuseColor)
                    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_EMISSION, self.emissionColor)
                    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, self.specularColor)

            if self.textureID is not None:
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureID)  # bind our texture to our shape

            if useCallLists:  # use call lists of original model, much faster for displaying
                # change openGL matrix to match model orientation and position
                GL.glPushMatrix()
                # matrix transformations steps: (applied in reverse order)
                # 1: move model initial joint (rotation) center to origin
                # 2: scale model
                # 3: rotate model to new orientation
                # 4: move model to parent joint position
#                 GL.glTranslatef(self.joint.location[0],self.joint.location[1],self.joint.location[2])
#                 GL.glMultMatrixf(numpy.array(q.toMat4()))
#                 scale = self.scale*self.joint.scale    #scale is combination of model scale and parent joint scale
#                 GL.glScalef(scale,scale,scale)
#                 GL.glTranslatef(-self.initialRotationCenter[0],-self.initialRotationCenter[1],-self.initialRotationCenter[2])

                if self.displayList is not None:
                    GL.glCallList(self.displayList)

                # go back to default matrix
                GL.glPopMatrix()
            else:  # don't use call lists, display recalculated vertexes
                # TODO: test how scaling works when vertexes are recalculated
                # TODO: Test how joint translation works
                GL.glBegin(GL.GL_TRIANGLES)
                for i in xrange(len(self.TriangleVertexIndexList)):
                    GL.glNormal3fv(self.NormVectors[i])
                    GL.glVertex3fv(self.VertexList[self.TriangleVertexIndexList[i, 0]])
                    GL.glVertex3fv(self.VertexList[self.TriangleVertexIndexList[i, 1]])
                    GL.glVertex3fv(self.VertexList[self.TriangleVertexIndexList[i, 2]])
                GL.glEnd()

    def _removeDuplicateVertices(self):
        # FIXME: not currently working
        origTriangleVertexIndexListShape = self.TriangleVertexIndexList.shape
        self.TriangleVertexIndexList = self.TriangleVertexIndexList.flatten()
        i = 0
        while i < len(self.OriginalVertexList) - 1:
            v = self.OriginalVertexList[i]
            sameValIndex = []
            for j in xrange(i + 1, len(self.OriginalVertexList)):
                if (v == self.OriginalVertexList[j]).all():
                    sameValIndex.append(j)
                    # change vertex index numbers in TriangleVertexIndexList
                    sameTriVertexIndex = numpy.where(self.TriangleVertexIndexList == j)
                    self.TriangleVertexIndexList[sameTriVertexIndex] = i
            # remove duplicate vertices
            self.OriginalVertexList = numpy.delete(self.OriginalVertexList, sameValIndex, 0)
            i += 1
        self.VertexList = self.OriginalVertexList.copy()
        # change shape of TriangleVertexIndexList back to original of Nx3
        self.TriangleVertexIndexList.shape = origTriangleVertexIndexListShape

    def updateFromJoint(self):
        # TODO: I think this has been replaced by the transformVertices function
        if self.joint is None:
            return
        if self.updateOnlyFromGrandparentJoints:
            if self.joint.parentJoint is None:
                q = quat(1, 0, 0, 0)
            else:
                q = self.joint.parentJoint.orientation
        else:
            q = self.joint.orientation
        # update vertex
        for i in xrange(len(self.OriginalVertexList)):
            scale = self.scale * self.joint.scale  # scale is combination of model scale factor and joint scale factor
            v = scale * (self.OriginalVertexList[i] - self.initialRotationCenter)  # get scaled vector based on initial rotation center
            v = numpy.array(q.rotateVec(v))  # apply current orientation
            self.VertexList[i] = v + self.joint.location  # move vector to joint location

        # update normals (orientation only, not scaled or moved)
        for i in xrange(len(self.OriginalNormVectors)):
            n = self.OriginalNormVectors[i]
            n = numpy.array(q.rotateVec(n))
            self.NormVectors[i] = n

    def calculateCenterOfGravity(self, recalculate=False):
        '''
        estimates CoG by sum(2*triangle area * triangle Cog) / sum(2*triangle area)
        from http://paulbourke.net/geometry/polyarea/
        assumes uniform density
        '''
        # TODO: I don't think this is used anymore
        if self.CoG is None or recalculate is True:
            t1 = time.time()
            num = 0.0
            den = 0.0
            for tri in self.TriangleVertexIndexList:
                centerVertex = (self.OriginalVertexList[tri[0]] + self.OriginalVertexList[tri[1]] + self.OriginalVertexList[tri[2]]) / 3
                v = numpy.cross(self.OriginalVertexList[tri[1]] - self.OriginalVertexList[tri[0]], self.OriginalVertexList[tri[2]] - self.OriginalVertexList[tri[0]])
                faceArea2x = numpy.sqrt(numpy.vdot(v, v))
                num += faceArea2x * centerVertex
                den += faceArea2x
            t2 = time.time()
            print "Time to cal CoG:", t2 - t1
            self.CoG = num / den
#             self.CoG = numpy.sum(self.OriginalVertexList) / len(self.OriginalVertexList)
        return self.CoG

    def PCA(self, recalculate=False):
        '''
        Perform Principle component analysis to get 3 major axis
        Equations from paper 'A robust mesh watermarking scheme based on PCA' by Bin Yang, Xiao-Qian Li, Wei Li
        '''
        if self.PCAvectors is None or recalculate is True:
#             vc = self.calculateCenterOfGravity()
#             #TODO: this might take up a lot of memory, check if its faster to do the offset all at once, or as we go
#             vertList = self.OriginalVertexList - vc
            vertList = self.OriginalVertexList.copy()

            C = numpy.cov(vertList, rowvar=0)
            d, v = numpy.linalg.eig(C)

            # ordering from greatest to least is not guaranteed, manually sort
            indx = numpy.argsort(-d)
            d = d[indx]
            v = v[:, indx]
            self.PCAvectors = v.T
            print 'Principle Component Analysis axis:'
            print self.PCAvectors
        return self.PCAvectors

    def compareToTransformedPoints(self, point, currentClosestSqDistance=None, currentModelID=None, modelName=''):
        # calculate closest distance from point to a vertex in this model
        # TODO: faster way? check to see distance to bounding box planes. If these are not better than currentClosestDistance, don't need to check every vertex of this model

        # check point distance to bounding box planes, if none of them is shorter than currentClosest distance, no need to keep looking at this bone
        bbPoints = numpy.array([[self.transformedMinPoint[0], self.transformedMinPoint[1], self.transformedMinPoint[2]],
                                [self.transformedMinPoint[0], self.transformedMinPoint[1], self.transformedMaxPoint[2]],
                                [self.transformedMinPoint[0], self.transformedMaxPoint[1], self.transformedMinPoint[2]],
                                [self.transformedMinPoint[0], self.transformedMaxPoint[1], self.transformedMaxPoint[2]],
                                [self.transformedMaxPoint[0], self.transformedMinPoint[1], self.transformedMinPoint[2]],
                                [self.transformedMaxPoint[0], self.transformedMinPoint[1], self.transformedMaxPoint[2]],
                                [self.transformedMaxPoint[0], self.transformedMaxPoint[1], self.transformedMinPoint[2]],
                                [self.transformedMaxPoint[0], self.transformedMaxPoint[1], self.transformedMaxPoint[2]]])
        placesPointNumbers = numpy.array([[0, 1, 3, 2],
                                          [0, 2, 6, 4],
                                          [0, 4, 5, 1],
                                          [2, 3, 7, 6],
                                          [1, 5, 7, 3],
                                          [4, 6, 7, 5]])

        if currentClosestSqDistance is not None and currentModelID is not None:
            for i in xrange(placesPointNumbers.shape[0]):
                v1 = bbPoints[placesPointNumbers[i, 1]] - bbPoints[placesPointNumbers[i, 0]]
                v2 = bbPoints[placesPointNumbers[i, 3]] - bbPoints[placesPointNumbers[i, 0]]
                v = numpy.cross(v1, v2)
                w = point - bbPoints[placesPointNumbers[i, 0]]
                distance = numpy.abs(numpy.dot(v, w)) / numpy.linalg.norm(v)
                if distance < currentClosestSqDistance:
                    break
            else:
                # no distance is less then currentClosest, just return
#                 print 'Skipped Bone %d %s' % (id(self), self.name)
                return currentClosestSqDistance, currentModelID, modelName

        # this bone might have a vertex closer, try looking
        minDistance = numpy.min(numpy.sqrt(numpy.sum((point - self.transformedVertexList) ** 2, axis=1)))
        if currentClosestSqDistance is None or minDistance < currentClosestSqDistance:
            currentClosestSqDistance = minDistance
            currentModelID = id(self)
            modelName = self.name
        return currentClosestSqDistance, currentModelID, modelName

    def compareToTransformedPointsKDTrees(self, point, currentClosestSqDistance=None, currentModelID=None, modelName=''):
        distance, i = self.kdtree.query(point)
        if currentClosestSqDistance is None or distance < currentClosestSqDistance:
            currentClosestSqDistance = distance
            currentModelID = id(self)
            modelName = self.name
        return currentClosestSqDistance, currentModelID, modelName


def createRectangularSolid(*args, **kwargs):
    '''
    Function Signatures:
    createRectangularSolid(...)
    createRectangularSolid(dimensions, ...)
    createRectangularSolid(dimensions, offset, ...)

    Arguments {default value}:

    dimensions
        array like object of form [length, width, height] that describes the
        dimensions of the model to be created. {1,1,1}
    offset
        array like object of form [x, y, z] that describes the
        offset applied to each vertex of the model. {0,0,0}

    Keywords:
    joint
        Joint object that the model will rotate around. {None}
    '''
    if len(args) > 0:
        dimensions = numpy.array(args[0], dtype=float)
    else:
        dimensions = numpy.array([1.0, 1.0, 1.0])

    x = dimensions[0]
    y = dimensions[1]
    z = dimensions[2]
    vList = []

    # z=0 plane
    f = numpy.array([[0.0, 0.0, 0.0],
                     [x, 0.0, 0.0],
                     [x, y, 0.0]])
    vList.extend(f)
    f = numpy.array([[0.0, 0.0, 0.0],
                     [x, y, 0.0],
                     [0.0, y, 0.0]])
    vList.extend(f)
    # z=z plane
    f = numpy.array([[x, y, z],
                     [x, 0.0, z],
                     [0.0, 0.0, z]])
    vList.extend(f)
    f = numpy.array([[0.0, y, z],
                     [x, y, z],
                     [0.0, 0.0, z]])
    vList.extend(f)
    # y=0 plane
    f = numpy.array([[0.0, 0.0, 0.0],
                     [0.0, 0.0, z],
                     [x, 0.0, z]])
    vList.extend(f)
    f = numpy.array([[0.0, 0.0, 0.0],
                     [x, 0.0, z],
                     [x, 0.0, 0.0]])
    vList.extend(f)
    # y=y plane
    f = numpy.array([[x, y, z],
                     [0.0, y, z],
                     [0.0, y, 0.0]])
    vList.extend(f)
    f = numpy.array([[x, y, 0.0],
                     [x, y, z],
                     [0.0, y, 0.0]])
    vList.extend(f)
    # x=0 plane
    f = numpy.array([[0.0, 0.0, 0.0],
                     [0.0, y, 0.0],
                     [0.0, y, z]])
    vList.extend(f)
    f = numpy.array([[0.0, 0.0, 0.0],
                     [0.0, y, z],
                     [0.0, 0, z]])
    vList.extend(f)
    # x=x plane
    f = numpy.array([[x, y, z],
                     [x, y, 0.0],
                     [x, 0.0, 0.0]])
    vList.extend(f)
    f = numpy.array([[x, 0.0, z],
                     [x, y, z],
                     [x, 0, 0.0]])
    vList.extend(f)

    # add offset
    if len(args) > 1:
        vList += numpy.array(args[1], dtype=float)
    triVert = numpy.array(range(len(vList)))
    triVert = triVert.reshape([len(vList) / 3, 3])
    if 'joint' in kwargs:
        rect = TriModel(vList, triVert, kwargs['joint'], **kwargs)
    else:
        rect = TriModel(vList, triVert, **kwargs)
    return rect


def createCylinder(*args, **kwargs):
    '''
    Function Signatures:
    createCylinder(...)
    createCylinder(length, ...)
    createCylinder(length, radius, ...)
    createCylinder(length, radius, offset, ...)

    Arguments {default value}:

    length
        length of cylinder along z axis {1.0}
    radius
        radius of cylinder ends {1.0}
    offset
        offsets the center of the bottom cylinder cap.
        Given in form [x,y,z]. { [0.0,0.0,0.0] }

    Keywords:
    joint
        Joint object that the model will rotate around. {None}
    ngon
        number of edges used to approximate the circle endcaps {10}
    '''
    if len(args) > 0:
        length = float(args[0])
    else:
        length = 1.0
    if len(args) > 1:
        r = float(args[1])
    else:
        r = 1.0
    if 'ngon' in kwargs:
        N = int(kwargs['ngon'])
    elif 'Ngon' in kwargs:
        N = int(kwargs['Ngon'])
    else:
        N = 10
    if N < 3:
        N = 3

    vList = []

    # draw end caps
    for n in xrange(1, N - 1):
        x1 = r * math.cos(2 * math.pi * n / N)
        y1 = r * math.sin(2 * math.pi * n / N)
        x2 = r * math.cos(2 * math.pi * (n + 1) / N)
        y2 = r * math.sin(2 * math.pi * (n + 1) / N)
        # draw top circle
        f = [[r, 0.0, length],
             [x1, y1, length],
             [x2, y2, length]]
        vList.extend(f)
        # draw bottom circle
        f = [[r, 0.0, 0.0],
             [x2, y2, 0.0],
             [x1, y1, 0.0]]
        vList.extend(f)

    # draw sides
    for n in xrange(0, N):
        x1 = r * math.cos(2 * math.pi * n / N)
        y1 = r * math.sin(2 * math.pi * n / N)
        if n == N - 1:
            x2 = r
            y2 = 0.0
        else:
            x2 = r * math.cos(2 * math.pi * (n + 1) / N)
            y2 = r * math.sin(2 * math.pi * (n + 1) / N)
        f = [[x1, y1, 0.0],
             [x2, y2, 0.0],
             [x2, y2, length]]
        vList.extend(f)
        f = [[x1, y1, 0.0],
             [x2, y2, length],
             [x1, y1, length]]
        vList.extend(f)

    vList = numpy.array(vList)
    # add offset
    if len(args) > 2:
        vList += numpy.array(args[2], dtype=float)
    triVert = numpy.array(range(len(vList)))
    triVert = triVert.reshape([len(vList) / 3, 3])
    return TriModel(vList, triVert, **kwargs)


def createCone(*args, **kwargs):
    '''
    Function Signatures:
    createCone(...)
    createCone(radius, ...)
    createCone(radius, height, ...)
    createCone(radius, height, offset, ...)

    Arguments {default value}:

    radius
        radius of cone {1.0}
    height
        height of cone {twice value as radius}
    offset
        offsets the center of the bottom cylinder cap.
        Given in form [x,y,z]. { [0.0,0.0,0.0] }

    Keywords:
    joint
        Joint object that the model will rotate around. {None}
    ngon
        number of edges used to approximate the circle bottom {10}
    axis
        axis this cone is parallel to, 'x', 'y', or 'z'
    '''
    if len(args) > 0:
        r = float(args[0])
    else:
        r = 1.0
    if len(args) > 1:
        height = float(args[1])
    else:
        height = r * 2.0
    if 'ngon' in kwargs:
        N = int(kwargs['ngon'])
    elif 'Ngon' in kwargs:
        N = int(kwargs['Ngon'])
    else:
        N = 10
    if N < 3:
        N = 3
    if 'axis' in kwargs:
        if kwargs['axis'].lower() == 'x':
            initialOrintation = quat(math.pi / 2, [0, 1, 0])
        elif kwargs['axis'].lower() == 'y':
            initialOrintation = quat(-math.pi / 2, [1, 0, 0])
        else:
            initialOrintation = quat(1)
    else:
        initialOrintation = quat(1)
    vList = []
    normList = []

    for n in xrange(N):
        x1 = r * math.cos(2 * math.pi * n / N)
        y1 = r * math.sin(2 * math.pi * n / N)
        x2 = r * math.cos(2 * math.pi * (n + 1) / N)
        y2 = r * math.sin(2 * math.pi * (n + 1) / N)
        # draw cone
        f = [[0.0, 0.0, height],
             [x2, y2, 0.0],
             [x1, y1, 0.0]]
        vList.extend(f)
        f = numpy.array(f)
        norm = numpy.cross(f[2] - f[0], f[1] - f[0])
        normList.append(norm)
        # draw circle
        f = [[0.0, 0.0, 0.0],
             [x1, y1, 0.0],
             [x2, y2, 0.0]]
        vList.extend(f)
        f = numpy.array(f)
        norm = numpy.cross(f[2] - f[0], f[1] - f[0])
        normList.append(norm)

    vList = numpy.array(vList)
    normList = numpy.array(normList)

    # change vertex to new orientation
    for i in xrange(len(vList)):
        vList[i] = numpy.array(initialOrintation.rotateVec(vList[i]))

    # add offset
    if len(args) > 2:
        vList += numpy.array(args[2], dtype=float)
    triVert = numpy.array(range(len(vList)))
    triVert = triVert.reshape([len(vList) / 3, 3])
    if 'joint' in kwargs:
        rect = TriModel(vList, triVert, kwargs['joint'], normalVectors=normList, **kwargs)
    else:
        rect = TriModel(vList, triVert, normalVectors=normList, **kwargs)
    return rect


def createTolRegion(angle, AN=9, RN=8, **kwargs):
    angleSteps = numpy.linspace(0, angle, AN)
    rotSteps = numpy.linspace(0.0, 2 * math.pi, RN, endpoint=False)
    v = []
    tri = []
    for ai in xrange(angleSteps.shape[0]):
        if ai == 0:  # first point
            v.append(numpy.array([0.0, 0.0, 1.0]))
            continue
        a = angleSteps[ai]

        for ri in xrange(rotSteps.shape[0]):
            r = rotSteps[ri]
            # generate new vertex
            vtemp = numpy.array([0.0, 0.0, 1.0])
            T = numpyTransform.rotation(r, [0, 0, 1], N=3) * numpyTransform.rotation(a, [1, 0, 0], N=3)
            vtemp = (T * numpy.matrix(vtemp).T).getA().squeeze()
            v.append(vtemp)
            vi = len(v) - 1
            if ai == 1:  # first ring is special case
                if ri == 0:  # first rotation, there are only two points, not enough to make a triangle
                    continue
                tri.append([0, vi, vi - 1])
                if ri == RN - 1:  # last triangle in first ring
                    tri.append([0, 1, RN])
            else:  # normal case
                if ri < RN - 1:
                    tri.append([vi, vi - RN, vi - RN + 1])
                else:
                    tri.append([vi, vi - RN, vi - RN - RN + 1])
                if ri > 0:  # first rotation point can only make one triangle
                    tri.append([vi, vi - 1, vi - RN])
                if ri == RN - 1:  # need to do that triangle that wasn't possible the on first rotation point
                    tri.append([vi, vi - RN + 1 - RN, vi - RN + 1])

    else:  # close shape on last ring
        v.append(numpy.array([0.0, 0.0, 0.0]))  # last point
        vi = len(v) - 1
        for ri in xrange(RN - 1):
            tri.append([vi, vi - RN + ri, vi - RN + 1 + ri])
        else:  # make last triangle
            tri.append([vi, vi - 1, vi - RN])

    v = numpy.array(v)
    tri = numpy.array(tri)

    # TODO: align to vec

    return TriModel(v, tri, **kwargs)
