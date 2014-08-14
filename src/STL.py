# -*- coding:utf-8 -*-
"""
Created on Aug 8, 2011

@author: grant
"""
import struct
import time
from TriModel import TriModel
import numpy


class Face:
    def __init__(self, strData=None):
        if strData is not None:
            self._unpackData(strData)
        else:
            self.normVector = (0.0, 0.0, 0.0)
            self.vertex = [(0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0)]
            self.byteCnt = 0.0

    def _unpackData(self, strData):
        data = struct.unpack('<12fH', strData)
        self.normVector = (data[0], data[1], data[2])
        self.vertex = [(data[3], data[4], data[5]),
                       (data[6], data[7], data[8]),
                       (data[9], data[10], data[11])]
        self.byteCnt = data[12]


def STLparse(fileToParse):
    header = fileToParse.read(5)
    if header.lower() == 'solid':
        # ASCII STL file
        header = fileToParse.readline()
        faces = list()
        i = 0
        while (True):
            line = fileToParse.readline().lower().strip().split()
            if line[0] == 'facet' and line[1] == 'normal':
                faces.append(Face())
                faces[i].normVector = (float(line[2]), float(line[3]), float(line[4]))
                vi = 0
            elif line[0] == 'vertex':
                faces[i].vertex[vi] = (float(line[1]), float(line[2]), float(line[3]))
                vi += 1
            elif line[0] == 'endloop':
                vi = 0
            elif line[0] == 'endfacet':
                i += 1
            elif line[0] == 'endsolid':
                break
        triNum = i
    else:
        # binary STL file
        header += fileToParse.read(75)
        try:
            triNum = struct.unpack('<I', fileToParse.read(4))
            triNum = triNum[0]
        except:
            print 'problem'

        faces = list()
        for i in xrange(0, triNum):
            try:
                data = fileToParse.read(50)
                faces.append(Face(data))
            except:
                print 'Error reading face'

    return header, triNum, faces


def readSTLfile(filepath, verbose=False, **kwargs):
    if verbose:
        start = time.clock()
    f = open(filepath, 'rb')
    header, triNum, faces = STLparse(f)
    f.close()
    if verbose:
        print "Time to Open %s: %0.6f" % (filepath, time.clock() - start)
        print header.strip()
        print 'Number of Triangles:', triNum

#    #list method
#    vertexList = []
#    triangleVertexIndexList = []
#    normVec = []
#    for face in faces:
#        vertexList.extend(face.vertex)
#        l=len(vertexList)
#        triangleVertexIndexList.append([l-3,l-2,l-1])
#        normVec.append(face.normVector)
#    model = TriModel(vertexList, triangleVertexIndexList, normalVectors=normVec, name=header.strip(),**kwargs)

    # numpy method, not really any faster then list method
    vertexList = numpy.zeros((len(faces) * 3, 3))
    triangleVertexIndexList = numpy.array(range(len(faces) * 3))
    triangleVertexIndexList.shape = (len(faces), 3)
    normVec = numpy.zeros((len(faces), 3))
    for i in xrange(len(faces)):
        vertexList[i * 3:i * 3 + 3, :] = faces[i].vertex
        normVec[i, :] = faces[i].normVector
    if 'joint' in kwargs:
        model = TriModel(vertexList, triangleVertexIndexList, kwargs['joint'], normalVectors=normVec, name=header.strip(), **kwargs)
    else:
        model = TriModel(vertexList, triangleVertexIndexList, normalVectors=normVec, name=header.strip(), **kwargs)
    if 'name' in kwargs:
        model.name = kwargs['name']

    return model


def saveSTLFile(model, filepath, binary=True):
    f = open(filepath, 'w')
    if binary:
        # FIXME: saving as binary is not working
        header = 80 * ' '
        if len(model.name) >= 5 and model.name[:5].lower() == 'solid':
            h = 'binary:' + model.name
            header[:len(h)] = h
        else:
            h = model.name
        header = h + header[len(h):]
        header = header[:80]
        f.write(struct.pack('<80s', str(header)))
#        f.write(header)
        f.write(struct.pack('<I', len(model.TriangleVertexIndexList)))
        for i in xrange(len(model.TriangleVertexIndexList)):
            norm = model.NormVectors[i]
            v0 = model.VertexList[model.TriangleVertexIndexList[i, 0]]
            v1 = model.VertexList[model.TriangleVertexIndexList[i, 1]]
            v2 = model.VertexList[model.TriangleVertexIndexList[i, 2]]
            f.write(struct.pack('<12fH', norm[0], norm[1], norm[2],
                                v0[0], v0[1], v0[2], v1[0], v1[1],
                                v1[2], v2[0], v2[1], v2[2], 0))
    else:
        f.write('solid ' + model.name + '\n')
        for i in xrange(len(model.TriangleVertexIndexList)):
            v0 = model.VertexList[model.TriangleVertexIndexList[i, 0]]
            v1 = model.VertexList[model.TriangleVertexIndexList[i, 1]]
            v2 = model.VertexList[model.TriangleVertexIndexList[i, 2]]
            f.write('facet normal %f %f %f\nouter loop\n' % (model.NormVectors[i, 0], model.NormVectors[i, 1], model.NormVectors[i, 2]))
            f.write('vertex %f %f %f\n' % (v0[0], v0[1], v0[2]))
            f.write('vertex %f %f %f\n' % (v1[0], v1[1], v1[2]))
            f.write('vertex %f %f %f\n' % (v2[0], v2[1], v2[2]))
            f.write('endloop\nendfacet\n')
        f.write('endsolid ' + model.name)

    f.close()
