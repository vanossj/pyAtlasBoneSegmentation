'''
Created on Feb 4, 2012

@author: Jeff
'''

import numpy, math


def rotationMat2Euler(rotMat):
    '''
    Conventions:
    Coord Syste: right hand
    rotMat = Ry*Rz*Rx
    ref: http://www.euclideanspace.com/maths/geometry/rotations/euler/index.htm

    rotMat must be a rotation matrix only, i.e. no scaling

    angles in radians
    return (angleXaxis, angleYaxis, angleZaxis)
    '''
    rotMat, tx, ty, tz, sx, sy, sz = decomposeMatrix(rotMat)
    if (rotMat[1, 0] > 0.998):  # singularity at north pole
        angley = math.atan2(rotMat[0, 2], rotMat[2, 2])
        anglez = math.pi / 2
        anglex = 0.0
    elif (rotMat[1, 0] < -0.998):  # singularity at south pole
        angley = math.atan2(rotMat[0, 2], rotMat[2, 2])
        anglez = -math.pi / 2
        anglex = 0.0
    else:
        angley = math.atan2(-rotMat[2, 0], rotMat[0, 0])
        anglex = math.atan2(-rotMat[1, 2], rotMat[1, 1])
        anglez = math.asin(rotMat[1, 0])
    return anglex, angley, anglez


def pointsInToleranceRange(points, Vec, angle, trans):
    '''
    assumptions
    primaryVec starts at origin
    length of primary vec defines how far out points can be
    trans is in form [x,y,z] where each element is +/- limit of movement in that direction
    '''
    pindx = numpy.zeros(points.shape[0], dtype=numpy.bool)

    bbPoints = numpy.array([[-trans[0], -trans[1], -trans[2]],
                            [-trans[0], -trans[1], trans[2]],
                            [-trans[0], trans[1], -trans[2]],
                            [-trans[0], trans[1], trans[2]],
                            [ trans[0], -trans[1], -trans[2]],
                            [ trans[0], -trans[1], trans[2]],
                            [ trans[0], trans[1], -trans[2]],
                            [ trans[0], trans[1], trans[2]]])
    VecLen = numpy.sqrt(numpy.sum(Vec ** 2))
    VecNorm = Vec / VecLen

    # if point is within tolerance box its ok
    pindx = numpy.logical_or(pindx, numpy.all(trans - numpy.abs(points) >= 0, axis=1))

    pvecbb = numpy.array([points[j] - bbPoints for j in xrange(points.shape[0])])
    pvecLen = numpy.sqrt(numpy.sum(pvecbb ** 2, axis=2))
    # pvec is on the corner of the tolerance box, its ok
    pindx = numpy.logical_or(pindx, numpy.any(pvecLen == 0.0, axis=1))

    pvecNorm = pvecbb / numpy.repeat(pvecLen[:, :, numpy.newaxis], 3, axis=2)
    vn = numpy.tile(VecNorm, pvecNorm.shape[0] * pvecNorm.shape[1]).reshape(pvecNorm.shape)
    dot = numpy.sum(vn * pvecNorm, axis=2)  # dot product VecNorm*pvecNorm
    pangle = numpy.arccos(dot)
    angleComp = pangle <= angle
    lenComp = pvecLen <= VecLen
    # if, for any corner of tolerance box, angle between vectors is less than max and pvec length is less than Vec length, point is ok
    pindx = numpy.logical_or(pindx, numpy.any(numpy.logical_and(angleComp, lenComp), axis=1))

    # if there exists a case where for one vertex, the angle is bad but the length is good and another vertex where the length is bad but the angle is good
    # then some point inside the box would work as an origin for pvec so this point must be reachable
    indx = numpy.logical_xor(angleComp, lenComp)
    pindx = numpy.logical_or(pindx, numpy.logical_and(numpy.any(numpy.logical_and(angleComp, indx), axis=1), numpy.any(numpy.logical_and(lenComp, indx), axis=1)))

    return pindx


def pointsInToleranceRange2(points, Vec, angle, trans):
    '''
    This is the same as pointsInToleranceRange() except that it looks through every vertex and therefore is much slower. 
    '''
    pindx = numpy.zeros(points.shape[0], dtype=numpy.bool)

    bbPoints = numpy.array([[-trans[0], -trans[1], -trans[2]],
                            [-trans[0], -trans[1], trans[2]],
                            [-trans[0], trans[1], -trans[2]],
                            [-trans[0], trans[1], trans[2]],
                            [ trans[0], -trans[1], -trans[2]],
                            [ trans[0], -trans[1], trans[2]],
                            [ trans[0], trans[1], -trans[2]],
                            [ trans[0], trans[1], trans[2]]])
    VecLen = numpy.linalg.norm(Vec)
    VecNorm = Vec / VecLen
    pointsInTolBox = 0
    pointsOnCornersOfTolBox = 0
    pointsReachableFromCorners = 0
    pointsReachableFromInsideTolBox = 0

    # find closest point on trans bounding box
    for i in xrange(points.shape[0]):
        point = points[i]

        # if point is within tolerance box its ok
        if numpy.all(trans - numpy.abs(point) >= 0):
            pindx[i] = True
            pointsInTolBox += 1
            continue
        pvecbb = point - bbPoints
        pvecLen = numpy.apply_along_axis(numpy.linalg.norm, 1, pvecbb)

        # pvec is on the corner of the tolerance box, its ok
        if numpy.any(pvecLen == 0.0):
            pindx[i] = True
            pointsOnCornersOfTolBox += 1
            continue
        pvecNorm = numpy.array([pvecbb[j] / pvecLen[j] for j in xrange(pvecbb.shape[0])])
        pangle = numpy.array([numpy.arccos(numpy.dot(VecNorm, pvecNorm[j])) for j in xrange(pvecNorm.shape[0])])
        angleComp = pangle <= angle
        lenComp = pvecLen <= VecLen
        # if, for any corner of tolerance box, angle between vectors is less than max and pvec length is less than Vec length, point is ok
        if numpy.any(numpy.logical_and(angleComp, lenComp)):
            pindx[i] = True
            pointsReachableFromCorners += 1
            continue
        # if there exists a case where for one vertex, the angle is bad but the length is good and another vertex where the length is bad but the angle is good
        # then some point inside the box would work as an origin for pvec so this point must be reachable
        indx = numpy.logical_xor(angleComp, lenComp)
        if numpy.any(angleComp[indx]) and numpy.any(lenComp[indx]):
            pindx[i] = True
            pointsReachableFromInsideTolBox += 1
            continue

#    print 'Points in Tolerance Box:', pointsInTolBox
#    print 'Points on Corner of Tolerance Box:', pointsOnCornersOfTolBox
#    print 'Points Reachable from Corners of Tolerance Box:', pointsReachableFromCorners
#    print 'Points Reachable from Inside Tolerance Box:', pointsReachableFromInsideTolBox

    return pindx


def pointsInBox(points, boxDiagPointA, boxDiagPointB, scale=1.0):
    diagVec = numpy.append(boxDiagPointA, boxDiagPointB)
    diagVec = diagVec.reshape((2, 3))
    boxMin = diagVec.min(axis=0)
    boxMax = diagVec.max(axis=0)
    pointsIndx = numpy.ones(points.shape[0], dtype=numpy.bool)
    boxCenter = (boxMax + boxMin) / 2
    boxMax = scale * (boxMax - boxCenter) + boxCenter
    boxMin = scale * (boxMin - boxCenter) + boxCenter

    bbPoints = numpy.array([[boxMin[0], boxMin[1], boxMin[2]],
                            [boxMin[0], boxMin[1], boxMax[2]],
                            [boxMin[0], boxMax[1], boxMin[2]],
                            [boxMin[0], boxMax[1], boxMax[2]],
                            [boxMax[0], boxMin[1], boxMin[2]],
                            [boxMax[0], boxMin[1], boxMax[2]],
                            [boxMax[0], boxMax[1], boxMin[2]],
                            [boxMax[0], boxMax[1], boxMax[2]]])
    facePointIndx = numpy.array([[0, 1, 3, 2],
                                 [0, 2, 6, 4],
                                 [0, 4, 5, 1],
                                 [2, 3, 7, 6],
                                 [1, 5, 7, 3],
                                 [4, 6, 7, 5]])

    # remove model data that is in front the neck
    for plane in xrange(facePointIndx.shape[0]):
        v1 = bbPoints[facePointIndx[plane, 3]] - bbPoints[facePointIndx[plane, 0]]
        v2 = bbPoints[facePointIndx[plane, 1]] - bbPoints[facePointIndx[plane, 0]]
        norm = numpy.cross(v1, v2)  # normal points inside
        planePoint = bbPoints[facePointIndx[plane, 0]]
        pointVecs = points - planePoint
        distance = numpy.dot(pointVecs, norm)
        pointsIndx[distance < 0] = False  # if distance is < 0 then point is on outside of box
    return pointsIndx


def transformPoints(transform, points):
    '''
    transform must be a 4x4 matrix/array
    points must be a Nx3 or Nx4 matrix/array
    '''
    points = numpy.array(points)
    ndim = points.ndim
    if ndim == 1:
        points = points[numpy.newaxis, :]
    A = numpy.matrix(transform)

    B = numpy.ones((points.shape[0], 4))
    B[:, :3] = points
    B = numpy.matrix(B).T

    transformedPoints = ((A * B).T).getA()
    transformedPoints = transformedPoints[:, :3]
    if ndim == 1:
        transformedPoints = transformedPoints.squeeze()
    return transformedPoints


def translation(*args):
    transMat = numpy.matrix(numpy.identity(4, dtype=numpy.double))
    if len(args) == 3:
        transMat[0, 3] = args[0]
        transMat[1, 3] = args[1]
        transMat[2, 3] = args[2]
    else:
        transMat[0, 3] = args[0][0]
        transMat[1, 3] = args[0][1]
        transMat[2, 3] = args[0][2]
    return transMat


def rotation(angle, axis, N=3):
    '''
    rotationMatrix generates a NxN rotation matrix (default 3x3)
    axis is of form [x,y,z]
    angle is in radians
    equation from http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    '''
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    axis = numpy.array(axis, dtype=numpy.double)
    axis /= math.sqrt(numpy.dot(axis, axis.conj()))  # normalize axis
    x = axis[0]
    y = axis[1]
    z = axis[2]
    rotMat = numpy.matrix(numpy.identity(N))
    rotMat[0, 0] = t * x * x + c
    rotMat[0, 1] = t * x * y - z * s
    rotMat[0, 2] = t * x * z + y * s
    rotMat[1, 0] = t * x * y + z * s
    rotMat[1, 1] = t * y * y + c
    rotMat[1, 2] = t * y * z - x * s
    rotMat[2, 0] = t * x * z - y * s
    rotMat[2, 1] = t * y * z + x * s
    rotMat[2, 2] = t * z * z + c
    return rotMat


def decomposeMatrix(m):
    m = numpy.matrix(m)
    # force m to be 4x4 matrix
    if m.shape != (4, 4):
        mt = numpy.matrix(numpy.identity(4))
        mt[:m.shape[0], :m.shape[1]] = m
        m = mt
    sx = numpy.linalg.norm(m[0:3, 0])
    sy = numpy.linalg.norm(m[0:3, 1])
    sz = numpy.linalg.norm(m[0:3, 2])

    # remove scaling
    srev = scaling([sx, sy, sz], N=4).I
    m *= srev

    return m[:3, :3], m[0, 3], m[1, 3], m[2, 3], sx, sy, sz


def axisAngleFromMatrix(rotMatrix, angleInDegrees=False):
    '''
    gives axis angle rotation for the passed rotation matrix
    return axis, angle
    axis is array of length 3
    equation from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
    '''

    rotMatrix, tx, ty, tz, sx, sy, sz = decomposeMatrix(rotMatrix)

    m = numpy.matrix(rotMatrix)
    if m.shape[0] < 3 or m.shape[1] < 3:
        raise Exception('Incorrect matrix size')
    angle = math.acos((m[0, 0] + m[1, 1] + m[2, 2] - 1) / 2)
    if angleInDegrees:
        angle = math.degrees(angle)
    if angle != 0:
        axis = [(m[2, 1] - m[1, 2]) / math.sqrt((m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2),
                (m[0, 2] - m[2, 0]) / math.sqrt((m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2),
                (m[1, 0] - m[0, 1]) / math.sqrt((m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2)]
    else:
        axis = [1.0, 0.0, 0.0]
    return axis, angle


def coordinateSystemConversionMatrix(currentAxes, newAxes, N=3):
    '''
    This system generates a NxN matrix defines the rigid rotation transform
    required to change currentAxis into newAxis, this transform can be used
    on points to transform them from the currentAxis coordinate system to
    the newAxis coordinate system

    equation from http://www.j3d.org/matrix_faq/matrfaq_latest.html
    Question 'Q40. How do I use matrices to convert one coordinate system to another?'
    '''
    currentAxes = numpy.matrix(currentAxes)
    newAxes = numpy.matrix(newAxes)

    M = newAxes * currentAxes.I
    if N is not None or (M.shape[0] != N and M.shape[1] != N):
        Mtmp = numpy.identity(N)
        for i in xrange(M.shape[0]):
            if i >= N:
                break
            for j in xrange(M.shape[1]):
                if j >= N:
                    break
                Mtmp[i, j] = M[i, j]
        M = numpy.matrix(Mtmp)
    return M


def scaling(scale, N=3):
    '''
    scale can be a single number or an array of three values [x, y, z]
    '''
    M = numpy.matrix(numpy.identity(N, dtype=numpy.double))
    scale = numpy.array(scale)
    if scale.ndim == 0:
        M[0, 0] = scale
        M[1, 1] = scale
        M[2, 2] = scale
    elif scale.ndim == 1:
        M[0, 0] = scale[0]
        M[1, 1] = scale[1]
        M[2, 2] = scale[2]
    return M


def findDistanceToNearestNeighbor(point, targetPointList):
    minDistance = numpy.min(numpy.sqrt(numpy.sum((point - targetPointList) ** 2, axis=1)))
    return minDistance


def findNearestNeighbor(point, targetPointList):
    minDistance = None
    for targetPoint in targetPointList:
        dist = numpy.linalg.norm(targetPoint - point)
        if minDistance is None or dist < minDistance:
            minDistance = dist
            nearestNeighbor = targetPoint
    return nearestNeighbor

if __name__ == '__main__':
    vec = numpy.array([-0.882986, -0.06795621, 0.46445417])
    vec2 = numpy.array([-0.81685764, -0.29567476, 0.49529792])

#    vec = numpy.array([1.0,2.0,3.0])
#    vec2 = numpy.array([-1.0,7.0,-2.0])

    vec /= numpy.linalg.norm(vec)
    vec2 /= numpy.linalg.norm(vec2)

    axis = numpy.cross(vec, vec2)
    angle = numpy.arccos(numpy.dot(vec, vec2))

    print 'Axis', axis
    print 'Angle', angle

    rotmat = rotation(angle, axis, N=4)

    vec3 = transformPoints(rotmat, vec)

    print 'Vec', vec
    print 'Vec2', vec2
    print 'Vec3', vec3
