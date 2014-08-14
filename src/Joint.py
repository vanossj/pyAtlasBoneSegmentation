# -*- coding:utf-8 -*-
"""
Created on Aug 8, 2011

@author: grant
"""
import math
import numpy
from OpenGL import GL, GLU
import TriModel
# TODO: try and remove cgkit, use numpy matrix instead
from cgkit.cgtypes import quat
import numpyTransform

class Joint:
    def __init__(self, *args, **kwargs):
        '''
        Function Signatures:
        Joint()
        Joint(location, ...)

        Arguments {default value}:

        location
            array like object of form [x,y,z] containing x, y, and z coordinates
            of this Joint. { [0,0,0] }

        Keywords:
        parentJoint
            Joint object of which this Joint is a child. {None}
        models
            TriModel object of list of TriModel objects that represent bones
            that are attached to this joint. { [] }
        name
            Name of joint
        initialOrientation
            quaternion (type cgkit.cgtypes.quat) representing the initial
            orientation. { quat(1,0,0,0) }
        showAxis
            Boolean that determines whether or not a 3D representation of the
            Joint is visible. { False }
        axisScale
            Number that determines the size of the drawn axis. Must be greater
            than 0. { 1.0 }
        '''
#        self.translateMat = cgtypes.mat4.identity()
        self.scaleMat = numpy.matrix(numpy.identity(4))
        self.rotateMat = numpy.matrix(numpy.identity(4))

        if len(args) > 0:
            self.location = numpy.array(args[0], dtype=float)
        else:
            self.location = numpy.array([0.0, 0.0, 0.0], dtype=float)
        self.initialLocationMat = numpy.matrix([[1.0, 0.0, 0.0, self.location[0]],
                                                [0.0, 1.0, 0.0, self.location[1]],
                                                [0.0, 0.0, 1.0, self.location[2]],
                                                [0.0, 0.0, 0.0, 1.0]])
        self.translateMat = numpy.matrix([[1.0, 0.0, 0.0, self.location[0]],
                                          [0.0, 1.0, 0.0, self.location[1]],
                                          [0.0, 0.0, 1.0, self.location[2]],
                                          [0.0, 0.0, 0.0, 1.0]])
        self.transformICP = numpy.matrix(numpy.identity(4))

        self.childJoints = []
#        self.scale = 1.0
        self.locationUnityScale = self.location.copy()
        self.type = 'ball'
        self.length = 10.0
        self.proximodistalVec = numpy.array([1.0, 0.0, 0.0])
        self.secondaryVec = numpy.array([0.0, 1.0, 0.0])
        self.tertiaryVec = numpy.array([0.0, 0.0, 1.0])
        self.proximodistalVecTransformed = numpy.array([1.0, 0.0, 0.0])
        self.secondaryVecTransformed = numpy.array([0.0, 1.0, 0.0])
        self.tertiaryVecTransformed = numpy.array([0.0, 0.0, 1.0])
        self.proximodistalVecScaled = self.length * self.proximodistalVec
        self.proximodistalVecTransformedScaled = self.length * self.proximodistalVecTransformed
        self.DOFvec = numpy.array([0, 0, 10.0])
        self.DOFangle = math.radians(45.0)
        self.DOFtrans = 5.0

        if 'parentJoint' in kwargs and isinstance(kwargs['parentJoint'], Joint):
            self.parentJoint = kwargs['parentJoint']
            self.parentJoint.childJoints.append(self)
        else:
            self.parentJoint = None

        self.models = []
        if 'models' in kwargs:
            try:
                for model in kwargs['models']:
                    if isinstance(model, TriModel):
                        self.models.append(model)
                        self.models[-1].setJoint(self)
            except TypeError:
                if isinstance(kwargs['models'], TriModel):
                    self.models.append(kwargs['models'])
                    self.models[-1].setJoint(self)

        if 'initialOrientation' in kwargs and isinstance(kwargs['initialOrientation'], quat):
            self.orientation = kwargs['initialOrientation']
        else:
            self.orientation = quat(1, 0, 0, 0)
        self.xAngle = 0.0
        self.yAngle = 0.0
        self.zAngle = 0.0

        if 'showAxis' in kwargs and isinstance(kwargs['showAxis'], bool):
            self.showAxis = kwargs['showAxis']
        else:
            self.showAxis = False
        if 'axisScale' in kwargs and kwargs['axisScale'] > 0.0:
            self.axisScale = kwargs['axisScale']
        else:
            self.axisScale = 1.0
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'Joint'

        for model in self.models:
            model.initialRotationCenter = self.location.copy()

        if self.parentJoint is None:
            self.initalLocationRelativeToParentJoint = self.location.copy()
            self.initialRelativeOrientationFromParent = self.orientation * quat(1, 0, 0, 0).inverse()
        else:
            self.initalLocationRelativeToParentJoint = self.location - self.parentJoint.location
            self.initialRelativeOrientationFromParent = self.orientation * self.parentJoint.orientation.inverse()
        self.relativeOrientationFromParent = quat(self.initialRelativeOrientationFromParent)

        self.createAxis(self.showAxis)

    def translate(self, coord, absolute=True):
        '''
        Arguements {Default value}
        coord
            array like object with dimensions 1x3 in format [x,y,z]
        absolute
            boolean that tells whether coord is the point to set joint to,
            or the step by with to move the joint from its current location
        '''
        coord = numpy.array(coord)
        if coord.shape != (3,):
            raise Exception("Incorrect input parameters")
        if absolute:
            self.translateMat = numpyTransform.translation(coord)
        else:
            self.translateMat *= numpyTransform.translation(coord)

    def rotate(self, *args, **kwargs):
        '''
        Function Signatures:
        rotate(q, ...)
        rotate(mat, ...)
        rotate(angle, axis, ...)
        rotate(xAngle, yAngle, zAngle, ...)
        rotate(elevation, azimuth, spin, sphericalCoord=True, ...)

        Arguments {default value}:
        q
            quaternion (cgkit.cgtypes.quat) that defines joint orientation
            (relative or absolute)
        mat
            4x4 rotation matrix (cgkit.cgtypes.mat4) that defines joint orientation
        angle
            angle (degrees or radians, radians default) which to rotate around
            axis
        axis
            vector [i,j,k] which defines axis to rotate around
        xAngle
            angle (degrees or radians, radians default) which to rotate around
            x axis
        yAngle
            angle (degrees or radians, radians default) which to rotate around
            y axis
        zAngle
            angle (degrees or radians, radians default) which to rotate around
            z axis
        elevation
            elevation angle (spherical coordinate system)
        azimuth
            azimuth angle (spherical coordinate system)
        spin
            spin angle around vector defined by elevation and azimuth

        Keywords:
        relative
            defines whether the change in orientation is relative or absolute
            {False}
        updateModels
            flag that determines if child models should be updated {True}
        angleOrder
            string that determines the order that Euler angle rotations are
            applied. {'xyz'}
        unitsDegrees
            flag that indicates what units the passed angles are in, degrees or
            radians. {False}
        sphericalCoord
            flag that indicates passed angles are spherical coordinates. In the
            spherical coordinate system, elevation rotates around the x axis
            first, then azimuth rotates around the y axis, finally spin rotates
            around the vector created by elevation and azimuth {False}
        '''

        if 'relative' in kwargs:
            relative = kwargs['relative']
        else:
            relative = False
        if 'updateModels' in kwargs:
            updateModels = kwargs['updateModels']
        else:
            updateModels = True
        if 'sphericalCoord' in kwargs:
            sphericalCoord = kwargs['sphericalCoord']
        else:
            sphericalCoord = False

        # the baseOrientation of the joint is either its current orientation, relativeAngle=True
        # or the baseOrientation is the initial relative orientation from the parent joint, relativeAngle=False
        if relative:
            baseOrientation = quat(self.orientation)
        # change by original orientation difference, to regain original orientation, relative to parent
        elif self.parentJoint is None:
            baseOrientation = self.initialRelativeOrientationFromParent * quat(1, 0, 0, 0)
        else:
            baseOrientation = self.initialRelativeOrientationFromParent * self.parentJoint.orientation

        if len(args) == 1:  # Quaternion rotation
            if isinstance(args[0], quat):
                rotateMat = args[0].toMat4()
            else:
                rotateMat = args[0]
        elif len(args) == 2:  # angle and axis rotation
            angle = args[0]
            axis = args[1]

            # convert angle units to radians if required
            if 'unitsDegrees' in kwargs and kwargs['unitsDegrees']:
                angle = math.radians(angle)
            angle = NormalizeAngleRad(angle)

            # create rotate matrix
            rotateMat = numpyTransform.rotation(angle, axis, N=4)

        elif len(args) == 3:  # Euler angle rotation
            xAngle = args[0]
            yAngle = args[1]
            zAngle = args[2]
            if 'angleOrder' in kwargs and not sphericalCoord:
                angleOrder = kwargs['angleOrder'].lower()
                if len(angleOrder) != 3 or angleOrder.find('x') < 0 or angleOrder.find('y') < 0 or angleOrder.find('z') < 0:
                    raise Exception('invalid angle order string')
            else:
                angleOrder = 'xyz'
            if 'unitsDegrees' in kwargs and kwargs['unitsDegrees']:
                xAngle = math.radians(xAngle)
                yAngle = math.radians(yAngle)
                zAngle = math.radians(zAngle)
            if 'relative' in kwargs and kwargs['relative']:
                self.xAngle += xAngle
                self.yAngle += yAngle
                self.zAngle += zAngle
            else:
                self.xAngle = xAngle
                self.yAngle = yAngle
                self.zAngle = zAngle
            if sphericalCoord:
                self.xAngle = NormalizeAngleRad(self.xAngle, -math.pi / 2, math.pi / 2, math.pi)
                self.yAngle = NormalizeAngleRad(self.yAngle)
                self.zAngle = NormalizeAngleRad(self.zAngle)
            else:
                self.xAngle = NormalizeAngleRad(self.xAngle)
                self.yAngle = NormalizeAngleRad(self.yAngle)
                self.zAngle = NormalizeAngleRad(self.zAngle)

            rotateMat = numpy.matrix(numpy.identity(4))
            # create orientation quaternion by multiplying rotations around local
            # x,y, and z axis
            # TODO: maybe flip the order of this rotation application?
            # FIXME: Spherical rotation not working
            for i in xrange(3):
                if angleOrder[i] == 'x':
                    rotateMat *= numpyTransform.rotation(self.xAngle, [1.0, 0.0, 0.0], N=4)
                if angleOrder[i] == 'y':
                    rotateMat *= numpyTransform.rotation(self.yAngle, [0.0, 1.0, 0.0], N=4)
                if angleOrder[i] == 'z':
                    rotateMat *= numpyTransform.rotation(self.zAngle, [0.0, 0.0, 1.0], N=4)

        else:  # invalid signature
            raise Exception("Invalid Function Signature")

        if relative:
            self.rotateMat *= rotateMat
        else:
            self.rotateMat = rotateMat

#    def setScale(self,*args,**kwargs):
#        #TODO:remove after mat4 transformation is done, also rename scaleTempname to scale, remote scale float value
#        self.scaleTempname(*args,**kwargs)

    def scale(self, *args, **kwargs):
        '''
        Arguments {Default value}
        scale(scale, ...)
        scale(scaleX,scaleY,scaleZ, ...)
        scale([scaleX,scaleY,scaleZ], ...)

        scale
            scale in the X,Y, and Z direction
        scaleX
            scale in the X dimension
        scaleY
            scale in the Y dimension
        scaleZ
            scale in the Z dimension

        keyword arguments
        absolute    {True}
            boolean that tells whether scale is the new scale (True)
            or an amount to adjust current scale by (False)
        '''
        if len(args) == 1:
            scale = numpy.array(args[0], dtype=float)
            if scale.shape != (3,):
                if scale.shape == ():
                    scale = numpy.repeat(scale, 3)
                else:
                    scale = numpy.repeat(scale[0], 3)
        elif len(args) == 3:
            scale = numpy.array([args[0], args[1], args[2]], dtype=float)
        if 'absolute' in kwargs and kwargs['absolute'] is False:
            self.scaleMat *= numpyTransform.scaling(scale, N=4)
        else:
            self.scaleMat = numpyTransform.scaling(scale, N=4)

    def createAxis(self, axisVisible):
        self.xAxis = TriModel.createCone(self.axisScale / 4, self.axisScale, self.location, name='axis_X', joint=self, axis='x', updateOnlyFromGrandparentJoints=True, visible=axisVisible, color=[1.0, 0.0, 0.0, 1.0])
        self.yAxis = TriModel.createCone(self.axisScale / 4, self.axisScale, self.location, name='axis_Y', joint=self, axis='y', updateOnlyFromGrandparentJoints=True, visible=axisVisible, color=[0.0, 1.0, 0.0, 1.0])
        self.zAxis = TriModel.createCone(self.axisScale / 4, self.axisScale, self.location, name='axis_Z', joint=self, axis='z', updateOnlyFromGrandparentJoints=True, visible=axisVisible, color=[0.0, 0.0, 1.0, 1.0])

    def OpenGLPaint(self, colorDrivenMaterial=None, useCallLists=True, parentTransform=numpy.matrix(numpy.identity(4))):
        # push matrix
        GL.glPushMatrix()

        # matrix transformations steps: (applied in reverse order)
        # 1: move model initial rotation center to origin
        # 2: scale model
        # 3: rotate model to new orientation
        # 4: move model to parent joint position
        if self.name == 'Neck':
            pass
        GL.glTranslatef(self.translateMat[0, 3], self.translateMat[1, 3], self.translateMat[2, 3])  # aka GL.glMultMatrixf(numpy.array(self.translateMat))
        GL.glMultMatrixf(numpy.array(self.rotateMat).T)  # need to transpose this because numpy matrices are row-major but OpenGl is expecting column-major matrix
#        axis, angle = numpyTransform.axisAngleFromMatrix(self.rotateMat, angleInDegrees=True)
#        GL.glRotated(angle, axis[0], axis[1], axis[2])
        GL.glScalef(self.scaleMat[0, 0], self.scaleMat[1, 1], self.scaleMat[2, 2])
        GL.glTranslatef(-self.initialLocationMat[0, 3], -self.initialLocationMat[1, 3], -self.initialLocationMat[2, 3])

#        if self.name == 'Neck':
#            print 'Neck Draw Transform'
#            print 'Rotation:'
#            print self.rotateMat
#            print 'Translation:'
#            print self.translateMat
#            print 'Scale'
#            print self.scaleMat
#            print 'Original location'
#            print self.initialLocationMat
#            print 'Transform'
#            tform = self.translateMat * self.rotateMat * self.scaleMat * self.initialLocationMat.I
#            print tform

        # draw models
        for model in self.models:
            model.OpenGLPaint(colorDrivenMaterial, useCallLists)
        # recursivly paint child joints
        for childJoint in self.childJoints:
            childJoint.OpenGLPaint(colorDrivenMaterial, useCallLists)
        # pop matrix
        GL.glPopMatrix()

    def transformVertices(self, baseTransform=numpy.matrix(numpy.identity(4)), modelID=None):
        # create transform matrix
        transform = numpy.matrix(baseTransform)
        transform *= self.translateMat
        transform *= self.rotateMat
        transform *= self.scaleMat
        transform *= self.initialLocationMat.I  #        transform *= numpyTransform.translation( (-self.initialLocationMat[0,3], -self.initialLocationMat[1,3], -self.initialLocationMat[2,3]) )

#        self.location = (transform * numpy.matrix([[self.locationUnityScale[0]], [self.locationUnityScale[1]], [self.locationUnityScale[2]], [1.0]])).getA().squeeze()[:3]
        self.location = numpyTransform.transformPoints(transform, self.locationUnityScale)

        transformRotScaleOnly = numpy.matrix(numpy.identity(4))
        transformRotScaleOnly[:3, :3] = transform[:3, :3]
        self.proximodistalVecTransformed = numpyTransform.transformPoints(transformRotScaleOnly, self.proximodistalVec[numpy.newaxis, :]).squeeze()
        self.proximodistalVecTransformedScaled = numpyTransform.transformPoints(transformRotScaleOnly, self.length * self.proximodistalVec[numpy.newaxis, :]).squeeze()
        self.secondaryVecTransformed = numpyTransform.transformPoints(transformRotScaleOnly, self.secondaryVec[numpy.newaxis, :]).squeeze()
        self.tertiaryVecTransformed = numpyTransform.transformPoints(transformRotScaleOnly, self.tertiaryVec[numpy.newaxis, :]).squeeze()

        for model in self.models:
            model.transformVertices(transform, modelID)

        for childJoint in self.childJoints:
            childJoint.transformVertices(transform, modelID)

    def getCummulativeTransform(self, jointID, baseTransform=numpy.matrix(numpy.identity(4))):
        transform = numpy.matrix(baseTransform)
        transform *= self.translateMat
        transform *= self.rotateMat
        transform *= self.scaleMat
        transform *= self.initialLocationMat.I

        if id(self) == jointID:
            return transform
        else:
            retTform = None
            for childJoint in self.childJoints:
                tempTform = childJoint.getCummulativeTransform(jointID, transform)
                if tempTform is not None:
                    retTform = tempTform
                    break
            return retTform

    def createKDTrees(self):
        for model in self.models:
            if model.visible and model.name[:5] != 'axis_':  # ignore models that are not visible and models that illustrate the axis
                model.createKDTrees()
        for childJoint in self.childJoints:
            childJoint.createKDTrees()

    def getBoundingBox(self, baseTransform=numpy.matrix(numpy.identity(4))):
        # TODO: change this to use lists and then use numpy to search for max and min along axis=0 instead of constantly comparing values
        points = []
        minPoint = None
        maxPoint = None

        # create transform matrix
        transform = numpy.matrix(baseTransform)
        transform *= self.translateMat
        transform *= self.rotateMat
        transform *= self.scaleMat
        transform *= self.initialLocationMat.I  #        transform *= numpyTransform.translation( (-self.initialLocationMat[0,3], -self.initialLocationMat[1,3], -self.initialLocationMat[2,3]) )

        for model in self.models:
            if model.visible:
                vMin2, vMax2 = model.getBoundingBox(transform)
                if vMin2 is not None and vMax2 is not None:
                    points.append(vMin2)
                    points.append(vMax2)

        for childJoint in self.childJoints:
            vMin2, vMax2 = childJoint.getBoundingBox(transform)
            if vMin2 is not None and vMax2 is not None:
                points.append(vMin2)
                points.append(vMax2)

        if len(points) > 1:
            points = numpy.array(points)
            minPoint = numpy.min(points, axis=0)
            maxPoint = numpy.max(points, axis=0)

        return minPoint, maxPoint

    def compareToTransformedPoints(self, point, currentClosestSqDistance=None, currentModelID=None, modelName=''):
        for model in self.models:
            if model.visible and model.name[:5] != 'axis_':  # ignore models that are not visible and models that illustrate the axis
                currentClosestSqDistance, currentModelID, modelName = model.compareToTransformedPoints(point, currentClosestSqDistance, currentModelID, modelName)
        for childJoint in self.childJoints:
            currentClosestSqDistance, currentModelID, modelName = childJoint.compareToTransformedPoints(point, currentClosestSqDistance, currentModelID, modelName)
        return currentClosestSqDistance, currentModelID, modelName

    def compareToTransformedPointsKDTrees(self, point, currentClosestSqDistance=None, currentModelID=None, modelName=''):
        for model in self.models:
            if model.visible and model.name[:5] != 'axis_':  # ignore models that are not visible and models that illustrate the axis
                currentClosestSqDistance, currentModelID, modelName = model.compareToTransformedPointsKDTrees(point, currentClosestSqDistance, currentModelID, modelName)
        for childJoint in self.childJoints:
            currentClosestSqDistance, currentModelID, modelName = childJoint.compareToTransformedPointsKDTrees(point, currentClosestSqDistance, currentModelID, modelName)
        return currentClosestSqDistance, currentModelID, modelName


def NormalizeAngleRad(angle, minimum=0.0, maximum=2 * math.pi, delta=2 * math.pi):
    while angle > maximum:
        angle -= delta
    while angle < minimum:
        angle += delta
    return angle

if __name__ == '__main__':
    pass
