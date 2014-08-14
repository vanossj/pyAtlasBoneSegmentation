'''
Created on Feb 29, 2012

@author: Jeff
'''
import mlabwrap
import os


class MatlabFmincon():
    def __init__(self):
        self.mlab = mlabwrap.init()
        self.mlab._do("clear all")
        self.mlab._do("cd('{}}')".format(os.getcwd()))
        self.mlab._do('global modelPoints dataPoints x0 lb ub')

    # def __del__(self):
    #     del self.mlab

    def minimize(self, modelPoints, dataPoints, x0, lb, ub, **kwargs):
        self.mlab._set('modelPoints', modelPoints.T)
        self.mlab._set('dataPoints', dataPoints.T)
        self.mlab._set('x0', x0)
        self.mlab._set('lb', lb)
        self.mlab._set('ub', ub)

        if 'scaleOnly' in kwargs and kwargs['scaleOnly']:
            self.mlab._do('[R, T, S] = mlabMinScale()')
        elif 'scaleOnlyIso' in kwargs and kwargs['scaleOnlyIso']:
            self.mlab._do('[R, T, S] = mlabMinScaleIso()')
        else:
            self.mlab._do('[R, T, S] = mlabMin()')

        R = self.mlab._get('R')
        T = self.mlab._get('T')
        S = self.mlab._get('S')

        return R, T, S
