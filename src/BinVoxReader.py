'''
Created on Oct 19, 2011

@author: Jeff
'''

import numpy, os, struct, time
from TriModel import TriModel


class BinVox:
    '''
    classdocs
    '''
    def __init__(self, filepath=None, verbose=False):
        self.scale = 1.0;
        self.translate = numpy.zeros(3)
        self.data = numpy.array([])
        self.dim = None
        self.model = None
        if filepath is not None:
            self.binvoxReader(filepath, verbose)

    def binvoxReader(self, filepath, verbose=False):
        if verbose:
            start = time.clock()
        fileName, fileExtension = os.path.splitext(filepath)  # @UnusedVariable
        bvFile = open(filepath, 'rb')
        if fileExtension == '.binvox':
            while True:  # first few lines are header info
                line = bvFile.readline()
                if line == '':
                    break
                tok = line.strip().split()
                if tok[0] == 'dim':
                    self.dim = numpy.array([int(tok[1]), int(tok[2]), int(tok[3])])
                if tok[0] == 'translate':
                    self.translate = numpy.array([float(tok[1]), float(tok[2]), float(tok[3])])
                if tok[0] == 'scale':
                    self.scale = float(tok[1])
                if tok[0] == 'data':
                    break
            # read data
            if self.dim is not None:
                numVoxels = self.dim[0] * self.dim[1] * self.dim[2]
                self.data = numpy.zeros(numVoxels)
                i = 0
                while i < numVoxels:
                    dStr = bvFile.read(2)
                    if dStr == '':
                        break
                    val, num = struct.unpack('BB', dStr)
                    self.data[i:i + num] = numpy.repeat(val, num)
                    i += num
                # resize data vector to 3D
                self.data.shape = (self.dim[0], self.dim[2], self.dim[1])
                # transpose data so indexing is (x,y,z)
                numpy.transpose(self.data, (0, 2, 1))
        bvFile.close()
        if verbose:
            print "Time to read %s: %0.6f" % (filepath, time.clock() - start)

    def createVoxelModel(self, iso=0.5, verbose=False):
        start = time.clock()
        vList = None
        triVert = None
        for x in xrange(self.data.shape[0]):
            for y in xrange(self.data.shape[1]):
                for z in xrange(self.data.shape[2]):
                    if self.data[x, y, z] < iso:
                        continue
                    vListTemp, triVertTemp = self._createBox(self.scale / self.dim, self.voxelLocation(x, y, z, toVoxelCenter=False))
                    if vList is None:
                        vList = vListTemp
                    else:
                        vList = numpy.concatenate((vList, vListTemp))
                    if triVert is None:
                        triVert = triVertTemp
                    else:
                        triVert = numpy.concatenate((triVert, triVertTemp + (triVert[-1, -1] + 1)))
        self.model = TriModel(vList, triVert)
        if verbose:
            print "Time to make model: %0.6f" % (time.clock() - start)
            print 'Number of Triangles:', len(self.model.TriangleVertexIndexList)
        return self.model

    def _createBox(self, *args):
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
        triVert = numpy.array([i for i in xrange(len(vList))])
        triVert = triVert.reshape([len(vList) / 3, 3])
        return vList, triVert

    def voxelLocation(self, i, j, k, toVoxelCenter=True):
        if self.dim is None:
            return numpy.zeros(3)
        if toVoxelCenter:
            offset = 0.5
        else:
            offset = 0.0
        pos = numpy.array(((i + offset) / self.dim[0], (j + offset) / self.dim[1], (k + offset) / self.dim[2]))
        return pos * self.scale + self.translate
