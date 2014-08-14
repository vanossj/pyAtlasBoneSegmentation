# -*- coding:utf-8 -*-
"""
Created on Aug 25, 2011

@author: grant
"""
import math, numpy
# TODO: try and remove cgkit, use numpy matrix instead
from cgkit.cgtypes import quat
from OpenGL import GL, GLU


class Camera:
    def __init__(self):
        self.cameraQuat = quat(1)  # identity quaternion
        self.camLoc = numpy.array([0.0, 0.0, 2.0])
        self.camFocus = numpy.array([0.0, 0.0, 0.0])
        self.camUp = numpy.array([0.0, 1.0, 0.0])

    def rotateAroundAxis(self, angle, axis):
        # create rotation quat based on local axis
        q = quat(angle, self.cameraQuat.rotateVec(axis))
        # rotate camera around focus point
        self.camLoc = numpy.array(q.rotateVec(self.camLoc - self.camFocus)) + self.camFocus
        # update camera orientation
        self.cameraQuat = q * self.cameraQuat
        # calculate up direction from new orientation
        self.camUp = numpy.array(self.cameraQuat.rotateVec([0.0, 1.0, 0.0]))

    def zoom(self, amount):
        '''amount is multiplied by distance, 0.5 is twice as close, 2.0 is twice as far'''
        if amount <= 0.0:
            return
        self.camLoc = (self.camLoc - self.camFocus) * amount + self.camFocus

    def pan(self, offset):
        '''offset must be in form [x,y,z]'''
        # change coordinates off offset to local orientation
        offset = self.cameraQuat.rotateVec(offset)
        self.camFocus += offset
        self.camLoc += offset

    def setView(self, *args, **kwargs):
        '''
        function signatures:
        setView(self, vMin, vMax, view='ortho1')
        setView(self, orientation, position, focusPoint):
        '''
        if isinstance(args[0], quat) and len(args) >= 3:
            self.cameraQuat = args[0]
            self.camLoc = numpy.array(args[1])
            self.camFocus = numpy.array(args[2])
            self.camUp = numpy.array(self.cameraQuat.rotateVec([0.0, 1.0, 0.0]))
        elif len(args) >= 2:
            vMin = numpy.array(args[0])
            vMax = numpy.array(args[1])
            view = 'ortho1'
            if 'view' in kwargs:
                view = kwargs['view']

            yAngle = None
            xAngle = None
            diameter = 2.0 * numpy.sqrt(numpy.sum((vMax - vMin) ** 2))
            self.camFocus = (vMax + vMin) / 2.0
            if view == 'top':
                yAngle = 0.0
                xAngle = -0.5 * math.pi
            elif view == 'bottom':
                yAngle = 0.0
                xAngle = 0.5 * math.pi
            elif view == 'right':
                yAngle = 0.5 * math.pi
                xAngle = 0.0
            elif view == 'left':
                yAngle = -0.5 * math.pi
                xAngle = 0.0
            elif view == 'front':
                yAngle = 0.0
                xAngle = 0.0
            elif view == 'back':
                yAngle = math.pi
                xAngle = 0.0
            elif view == 'ortho1':
                yAngle = 0.25 * math.pi
                xAngle = -0.25 * math.pi
            elif view == 'ortho2':
                yAngle = 0.75 * math.pi
                xAngle = -0.25 * math.pi
            elif view == 'ortho3':
                yAngle = 1.25 * math.pi
                xAngle = -0.25 * math.pi
            elif view == 'ortho4':
                yAngle = 1.75 * math.pi
                xAngle = -0.25 * math.pi
            elif view == 'ortho5':
                yAngle = 0.75 * math.pi
                xAngle = 0.25 * math.pi
            elif view == 'ortho6':
                yAngle = 1.25 * math.pi
                xAngle = 0.25 * math.pi
            elif view == 'ortho7':
                yAngle = 1.75 * math.pi
                xAngle = 0.25 * math.pi
            elif view == 'ortho8':
                yAngle = 0.25 * math.pi
                xAngle = 0.25 * math.pi

            if yAngle is not None and xAngle is not None:
                # create quat as rotation around y then around (local)x
                startQuat = quat(yAngle, [0, 1, 0])
#                startQuat = quat(numpy.pi, startQuat.rotateVec([0,1,0]))*startQuat
                self.cameraQuat = quat(xAngle, startQuat.rotateVec([1, 0, 0])) * startQuat
                self.camLoc = self.cameraQuat.rotateVec([0.0, 0.0, diameter])
                self.camLoc += self.camFocus
                self.camUp = numpy.array(self.cameraQuat.rotateVec([0.0, 1.0, 0.0]))

    def openGLTransform(self):
        GLU.gluLookAt(self.camLoc[0], self.camLoc[1], self.camLoc[2],
                      self.camFocus[0], self.camFocus[1], self.camFocus[2],
                      self.camUp[0], self.camUp[1], self.camUp[2])


class OpenGlLight:
    def __init__(self, lightNumber, enabled=False):
        self.enabled = enabled
        self.lightNumber = lightNumber
        self.specularColor = [0.0, 0.0, 0.0, 0.0]
        self.diffuseColor = [0.0, 0.0, 0.0, 1.0]
        self.ambientColor = [0.0, 0.0, 0.0, 1.0]
        self.emissiveColor = [0.0, 0.0, 0.0, 0.0]
        self.position = [0.0, 0.0, 1.0, 0.0]
        self.directional = True

    def updateOpenGl(self):
        if self.enabled:
            GL.glEnable(GL.GL_LIGHT0 + self.lightNumber)

            GL.glLightfv(GL.GL_LIGHT0 + self.lightNumber, GL.GL_DIFFUSE, self.diffuseColor)
            GL.glLightfv(GL.GL_LIGHT0 + self.lightNumber, GL.GL_AMBIENT, self.ambientColor)
            GL.glLightfv(GL.GL_LIGHT0 + self.lightNumber, GL.GL_SPECULAR, self.specularColor)
            if self.directional:
                self.position[3] = 0.0
            else:
                self.position[3] = 1.0
            GL.glLightfv(GL.GL_LIGHT0 + self.lightNumber, GL.GL_POSITION, self.position)
        else:
            GL.glDisable(GL.GL_LIGHT0 + self.lightNumber)
