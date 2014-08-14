'''
Created on Feb 15, 2012

@author: Jeff
'''
import numpy
import numpyTransform
from scipy.spatial import cKDTree as KDTree
# from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import scipy.optimize
import time
from math import pi
from MatlabFunctions import MatlabFmincon
import nlopt
import sys


class ICP(object):
    '''
    classdocs
    '''
    def __init__(self, modelPointCloud, dataPointCloud, **kwargs):
        '''
        Supported Signatures
        modelPointCloud
            The model point cloud is the base to which the data point cloud will be matched
        dataPointCloud
            The data point cloud is transformed so that it matches the model point cloud

        Key Word Arguments:
        maxIterations
            maximum number of iterations to perform, default is 10
            TODO: in the future provide an option to also account for minimum acceptable error
        matchingMethod
            'kdtree'        Use a KD-Tree for nearest neighbor search {default}
            'bruteforce'    Use brute force for nearest neighbor search
        minimizeMethod
            'point'            Use point to point minimization {default}
            'plane'            Use point to plane minimization
        weightMethod
            function that takes indices into the modelPointCloud and returns the weight of those indices
            By default all points are weighted equally
        modelDownsampleFactor
            integer that represents uniform sampling of model point cloud
            1 is no resampling, 2 is every other point, 3 is every third point...
        dataDownsampleFactor
            integer that represents uniform sampling of model point cloud
            1 is no resampling, 2 is every other point, 3 is every third point...

        ICP Process is five steps
            1: Input Filter
            2: Match
            3: Outlier Filter
            4: Error Minimization
            5: Check if error is less than limits
                yes: we are don
                no: go back to step 2 with new transformation function
        '''
        self.startTime = time.time()

        if 'modelDownsampleFactor' in kwargs and int(kwargs['modelDownsampleFactor']) > 1:
            factor = int(kwargs['modelDownsampleFactor'])
            temp = numpy.zeros(factor, dtype=numpy.bool)
            temp[-1] = True
            modelDownSampleIndices = numpy.tile(temp, (modelPointCloud.shape[0] / factor) + 1)[:modelPointCloud.shape[0]]
        else:
            modelDownSampleIndices = numpy.ones(modelPointCloud.shape[0], dtype=numpy.bool)
        if 'dataDownsampleFactor' in kwargs and int(kwargs['dataDownsampleFactor']) > 1:
            factor = int(kwargs['dataDownsampleFactor'])
            temp = numpy.zeros(factor, dtype=numpy.bool)
            temp[-1] = True
            dataDownSampleIndices = numpy.tile(temp, (dataPointCloud.shape[0] / factor) + 1)[:dataPointCloud.shape[0]]
        else:
            dataDownSampleIndices = numpy.ones(dataPointCloud.shape[0], dtype=numpy.bool)

        # TODO: uniform sampling of point clouds
        self.q = modelPointCloud[modelDownSampleIndices]
        self.p = dataPointCloud[dataDownSampleIndices]
        self.matlab = None

        # get kwargs
        if 'maxIterations' in kwargs:
            self.K = int(kwargs['maxIterations'])
        else:
            self.K = 10
        if 'matchingMethod' in kwargs:
            if kwargs['matchingMethod'] == 'bruteforce':
                self.matching = self.matchingBruteForce
            else:
                self.matching = self.matchingKDTree
                self.qKDTree = KDTree(self.q)
        else:
            self.matching = self.matchingKDTree
            self.qKDTree = KDTree(self.q)

        if 'minimizeMethod' in kwargs:
            if kwargs['minimizeMethod'] == 'plane':  # point to plane
                self.minimize = self.minimizePlane
            elif kwargs['minimizeMethod'] == 'fmincon':
                self.minimize = self.minimizeMatlab
                self.matlab = MatlabFmincon()
            elif kwargs['minimizeMethod'] == 'custom':
                self.minimize = self.minimizeCustom
            else:  # point to point
                self.minimize = self.minimizePoint
        else:
            self.minimize = self.minimizePoint

        if 'weightMethod' in kwargs:
            self.weightMethod = kwargs['weightMethod']
        else:
            self.weightMethod = self.weightEqual

        # initialize translation and rotation matrix
        self.transformMatrix = numpy.matrix(numpy.identity(4))
        # initialize list of translations and rotation matrix for each iteration of ICP
        self.totalTransformMatrix = [numpy.matrix(numpy.identity(4))]

        self.pt = self.p.copy()  # transformed point cloud
        self.t = []  # array of times for each iteration of ICP
        self.err = []  # error for each iteration of ICP
        self.Np = self.p.shape[0]  # number of points in data cloud

        # preprocessing finish, log time
        self.t.append(time.time() - self.startTime)
        print 'Time for preprocessing:', self.t[-1]

    def __del__(self):
        if self.matlab is not None:
            del self.matlab

    def runICP(self, **kwargs):
        tStart = time.time()

        # get 'global' tolerances
        if 'x0' in kwargs:
            kwargs['initX0'] = kwargs['x0'].copy()
        if 'lb' in kwargs:
            kwargs['initLB'] = kwargs['lb'].copy()
        if 'ub' in kwargs:
            kwargs['initUB'] = kwargs['ub'].copy()

        # main ICP loop
        for k in xrange(self.K):
            t1 = time.time()
            minDistances, nearestNeighbor = self.matching(self.pt)

            # get indices of the points we are interested in
            p_idx = numpy.ones(self.p.shape[0], dtype=numpy.bool)  # since there are no edges we are interested in all the points
            q_idx = nearestNeighbor
            print '\tTime to calc min distance:', time.time() - t1

            # TODO: Input filtering
            # reject some % of worst matches
            # Multiresolution sampling

            # add error for first iteration
            if k == 0:
                t1 = time.time()
                self.err.append(numpy.sqrt(numpy.sum(minDistances ** 2) / minDistances.shape[0]))
                print '\tInitial RMS error: %f, Time to calc: %f' % (self.err[-1], time.time() - t1)

            # generate rotation matrix and translation
            t1 = time.time()
            weights = self.weightMethod(nearestNeighbor)

            # get current cumulative rotation/translation in independent variable values, this way we can change the iteration bounds so that the global bounds are not violated
            cummulativeX0 = numpy.zeros(9)
            rotMat, tx, ty, tz, sx, sy, sz = numpyTransform.decomposeMatrix(self.totalTransformMatrix[-1])
            rx, ry, rz = numpyTransform.rotationMat2Euler(rotMat)
            cummulativeX0[0] = rx
            cummulativeX0[1] = ry
            cummulativeX0[2] = rz
            cummulativeX0[3] = tx
            cummulativeX0[4] = ty
            cummulativeX0[5] = tz
            cummulativeX0[6] = sx
            cummulativeX0[7] = sy
            cummulativeX0[8] = sz

            R, T, S = self.minimize(self.q[q_idx], self.pt[p_idx], weights=weights, cummulativeX0=cummulativeX0, **kwargs)
            print '\tTime to calc new transformation:', time.time() - t1

            # create combined transformation matrix, apply this relative transformation to current transformation
            transformMatrix = numpy.matrix(numpy.identity(4))
            transformMatrix *= T
            transformMatrix *= R
            transformMatrix *= S
            self.totalTransformMatrix.append(self.totalTransformMatrix[-1] * transformMatrix)

            # apply last transformation
            t1 = time.time()
            self.pt = numpyTransform.transformPoints(self.totalTransformMatrix[-1], self.p)
            print '\tTime to applying transform to all points:', time.time() - t1

            # root mean of objective function
            t1 = time.time()
            self.err.append(self.rms_error(self.q[q_idx], self.pt[p_idx]))
            print '\tIteration %d RMS error: %f, Time to calc: %f' % (k + 1, self.err[-1], time.time() - t1)

            # TODO: add extrapolation

            # store time to get to this iteration
            self.t.append(time.time() - self.startTime)
            print 'Iteration %d took %7.3f seconds' % (k + 1, self.t[-1] - self.t[-2])

        print 'Total ICP run time:', time.time() - tStart
        return self.totalTransformMatrix, self.err, self.t

    def matchingKDTree(self, points):
        minDistances, nearestNeighborIndex = self.qKDTree.query(points)
        return minDistances, nearestNeighborIndex

    def matchingBruteForce(self, points):
        nearestNeighborIndex = numpy.zeros(points.shape[0])
        distances = cdist(points, self.q)  # calculate all combination of point distances
        minDistances = distances.min(axis=1)
        for i in xrange(points.shape[0]):
            nearestNeighborIndex[i] = numpy.where(distances[i] == minDistances[i])[0][0]
        return minDistances, nearestNeighborIndex

    def minimizePoint(self, q, p, **kwargs):
        R = numpy.matrix(numpy.identity(4))
        T = numpy.matrix(numpy.identity(4))
        S = numpy.matrix(numpy.identity(4))

        if 'weights' in kwargs:
            weights = kwargs['weights']
        else:
            raise Warning('weights argument not supplied')
            return R, T
#        function [R,T] = eq_point(q,p,weights)
        m = p.shape[0]
        n = q.shape[0]

        # normalize weights
        weights = weights / weights.sum()

        # find data centroid and deviations from centroid
        q_bar = (numpy.mat(q.T) * numpy.mat(weights[:, numpy.newaxis])).getA().squeeze()
        q_mark = q - numpy.tile(q_bar, n).reshape((n, 3))
        # Apply weights
        q_mark = q_mark * numpy.repeat(weights, 3).reshape((weights.shape[0], 3))

        # find data centroid and deviations from centroid
        p_bar = (numpy.mat(p.T) * numpy.mat(weights[:, numpy.newaxis])).getA().squeeze()
        p_mark = p - numpy.tile(p_bar, m).reshape((m, 3))
        # Apply weights
        # p_mark = p_mark * numpy.repeat(weights, 3).reshape((weights.shape[0],3))

        N = (numpy.mat(p_mark).T * numpy.mat(q_mark)).getA()  # taking points of q in matched order

        [U, Ss, V] = numpy.linalg.svd(N);  # singular value decomposition
        V = (numpy.mat(V).H).getA()

        RMattemp = numpy.mat(V) * numpy.mat(U).T

        Ttemp = (numpy.mat(q_bar).T - RMattemp * numpy.mat(p_bar).T).getA().squeeze()

        R[:3, :3] = RMattemp.getA()
        T = numpyTransform.translation(Ttemp)

        return R, T, S

    def minimizeMatlab(self, modelPoints, dataPoints, **kwargs):
        if 'x0' in kwargs:
            x0 = kwargs['x0']
        else:
            raise Exception('There are no variables to solve for')

        # check for initial settings and bounds so that we can calculate current settings and bounds
        if 'initX0' in kwargs:
            initX0 = kwargs['initX0']
        if 'initLB' in kwargs:
            initLB = kwargs['initLB']
        if 'initUB' in kwargs:
            initUB = kwargs['initUB']
        if 'cummulativeX0' in kwargs:
            cummulativeX0 = kwargs['cummulativeX0']

        # NOTE: I think this only works if x0/initX) is all zeros
        ub = initUB - (cummulativeX0 - initX0)
        lb = initLB - (cummulativeX0 - initX0)
        # rounding errors can cause Bounds to be incorrect
        i = ub < x0
        if numpy.any(i):
            print 'upper bounds less than x0'
        ub[i] = x0[i] + 10 * numpy.spacing(x0[i])
        i = lb > x0
        if numpy.any(i):
            print 'lower bounds less than x0'
        lb[i] = x0[i] - 10 * numpy.spacing(x0[i])


#        if x0.shape[0] > 6 or ('scaleOnly' in kwargs and kwargs['scaleOnly']):
#            raise Exception('Scaling is not currently supported it will screw things up. Need some way to control scaling bounds so that it stays in global scaling bounds')
        try:
            if 'scaleOnly' in kwargs:
                R, T, S = self.matlab.minimize(modelPoints, dataPoints, x0[-3:], lb[-3:], ub[-3:], scaleOnly=kwargs['scaleOnly'])
            elif 'scaleOnlyIso' in kwargs:
                R, T, S = self.matlab.minimize(modelPoints, dataPoints, x0[-1:], lb[-1:], ub[-1:], scaleOnlyIso=kwargs['scaleOnlyIso'])
            else:
                R, T, S = self.matlab.minimize(modelPoints, dataPoints, x0[:6], lb[:6], ub[:6])  # only rotation and translation
        except:
            sys.stderr.write('ERROR: Problem with matlab, closing matlab\n')
            del self.matlab
            self.matlab = None

        return R, T, S

    def minimizeCustom(self, p, q, **kwargs):
        S = numpy.matrix(numpy.identity(4))
        # TODO: try using functions from the nlopt module

        def objectiveFunc(*args, **kwargs):
            d = p
            m = q
            params = args[0]
            if args[1].size > 0:  # gradient
                args[1][:] = numpy.array([pi / 100, pi / 100, pi / 100, 0.01, 0.01, 0.01])  # arbitrary gradient

#            transform = numpy.matrix(numpy.identity(4))
            translate = numpyTransform.translation(params[3:6])
            rotx = numpyTransform.rotation(params[0], [1, 0, 0], N=4)
            roty = numpyTransform.rotation(params[1], [0, 1, 0], N=4)
            rotz = numpyTransform.rotation(params[2], [0, 0, 1], N=4)
            transform = translate * rotx * roty * rotz

            Dicp = numpyTransform.transformPoints(transform, d)

#            err = self.rms_error(m, Dicp)
            err = numpy.mean(numpy.sqrt(numpy.sum((m - Dicp) ** 2, axis=1)))
#            err = numpy.sqrt(numpy.sum((m - Dicp) ** 2, axis=1))
            return err

        x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if 'optAlg' in kwargs:
            opt = nlopt.opt(kwargs['optAlg'], 6)
        else:
            opt = nlopt.opt(nlopt.GN_CRS2_LM, 6)

        opt.set_min_objective(objectiveFunc)
        opt.set_lower_bounds([-pi, -pi, -pi, -3.0, -3.0, -3.0])
        opt.set_upper_bounds([pi, pi, pi, 3.0, 3.0, 3.0])
        opt.set_maxeval(1500)
        params = opt.optimize(x0)

#        output = scipy.optimize.leastsq(objectiveFunc, x0, args=funcArgs)
#        params = output[0]

#        params = scipy.optimize.fmin(objectiveFunc, x0, args=funcArgs)

#        constraints = []
#        varBounds = [(-pi, pi), (-pi, pi), (-pi, pi), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)]
#        params = scipy.optimize.fmin_slsqp(objectiveFunc, x0, eqcons=constraints, bounds=varBounds, args=funcArgs)

#        output = scipy.optimize.fmin_l_bfgs_b(objectiveFunc, x0, bounds=varBounds, args=funcArgs, approx_grad=True)
#        params = output[0]
#        print  'Min error:', output[1]

#        params = scipy.optimize.fmin_tnc(objectiveFunc, x0, bounds=varBounds, args=funcArgs, approx_grad=True)
#        params = scipy.optimize.fmin_slsqp(objectiveFunc, x0, eqcons=constraints, bounds=varBounds, args=funcArgs)
#        params = scipy.optimize.fmin_slsqp(objectiveFunc, x0, eqcons=constraints, bounds=varBounds, args=funcArgs)

        translate = numpyTransform.translation(params[3:6])
        rotx = numpyTransform.rotation(params[0], [1, 0, 0], N=4)
        roty = numpyTransform.rotation(params[1], [0, 1, 0], N=4)
        rotz = numpyTransform.rotation(params[2], [0, 0, 1], N=4)
        transform = translate * rotx * roty * rotz
        return rotx * roty * rotz, S

    def minimizePlane(self, p, q, **kwargs):
        # TODO: Actually fill out
        R = numpy.matrix(numpy.identity(4))
        T = numpy.matrix(numpy.identity(4))
        S = numpy.matrix(numpy.identity(4))

#        function [R,T] = eq_plane(q,p,n,weights)
#        n = n .* repmat(weights,3,1);
#
#        c = cross(p,n);
#
#        cn = vertcat(c,n);
#
#        C = cn*transpose(cn);
#
#        b = - [sum(sum((p-q).*repmat(cn(1,:),3,1).*n));
#               sum(sum((p-q).*repmat(cn(2,:),3,1).*n));
#               sum(sum((p-q).*repmat(cn(3,:),3,1).*n));
#               sum(sum((p-q).*repmat(cn(4,:),3,1).*n));
#               sum(sum((p-q).*repmat(cn(5,:),3,1).*n));
#               sum(sum((p-q).*repmat(cn(6,:),3,1).*n))];
#
#        X = C\b;
#
#        cx = cos(X(1)); cy = cos(X(2)); cz = cos(X(3));
#        sx = sin(X(1)); sy = sin(X(2)); sz = sin(X(3));
#
#        R = [cy*cz cz*sx*sy-cx*sz cx*cz*sy+sx*sz;
#             cy*sz cx*cz+sx*sy*sz cx*sy*sz-cz*sx;
#             -sy cy*sx cx*cy];
#
#        T = X(4:6);

        return R, T, S

    def weightEqual(self, qIndices):
        return numpy.ones(qIndices.shape[0])

    def rms_error(self, a, b):
        '''
        Determine the RMS error between two point equally sized point clouds with point correspondence.
        NOTE: a and b need to have equal number of points
        '''
        if a.shape[0] != b.shape[0]:
            raise Exception('Input Point clouds a and b do not have the same number of points')

        distSq = numpy.sum((a - b) ** 2, axis=1)
        err = numpy.sqrt(numpy.mean(distSq))
        return err


def demo(*args, **kwargs):
    import math
    m = 80  # width of grid
    n = m ** 2  # number of points

    minVal = -2.0
    maxVal = 2.0
    delta = (maxVal - minVal) / (m - 1)
    X, Y = numpy.mgrid[minVal:maxVal + delta:delta, minVal:maxVal + delta:delta]

    X = X.flatten()
    Y = Y.flatten()

    Z = numpy.sin(X) * numpy.cos(Y)

    # Create the data point-matrix
    M = numpy.array([X, Y, Z]).T

    # Translation values (a.u.):
    Tx = 0.5
    Ty = -0.3
    Tz = 0.2

    # Translation vector
    T = numpyTransform.translation(Tx, Ty, Tz)

    S = numpyTransform.scaling(1.0, N=4)

    # Rotation values (rad.):
    rx = 0.3
    ry = -0.2
    rz = 0.05

    Rx = numpy.matrix([[1, 0, 0, 0],
                      [0, math.cos(rx), -math.sin(rx), 0],
                      [0, math.sin(rx), math.cos(rx), 0],
                      [0, 0, 0, 1]])

    Ry = numpy.matrix([[math.cos(ry), 0, math.sin(ry), 0],
                       [0, 1, 0, 0],
                       [-math.sin(ry), 0, math.cos(ry), 0],
                       [0, 0, 0, 1]])

    Rz = numpy.matrix([[math.cos(rz), -math.sin(rz), 0, 0],
                       [math.sin(rz), math.cos(rz), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    # Rotation matrix
    R = Rx * Ry * Rz

    transformMat = numpy.matrix(numpy.identity(4))
    transformMat *= T
    transformMat *= R
    transformMat *= S

    # Transform data-matrix plus noise into model-matrix
    D = numpyTransform.transformPoints(transformMat, M)

    # Add noise to model and data
    M = M + 0.01 * numpy.random.randn(n, 3)
    D = D + 0.01 * numpy.random.randn(n, 3)

    # Run ICP (standard settings)
    initialGuess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lowerBounds = numpy.array([-pi, -pi, -pi, -100.0, -100.0, -100.0])
    upperBounds = numpy.array([pi, pi, pi, 100.0, 100.0, 100.0])
    icp = ICP(M, D, maxIterations=15, dataDownsampleFactor=1, minimizeMethod='fmincon', **kwargs)
#    icp = ICP(M, D, maxIterations=15, dataDownsampleFactor=1, minimizeMethod='point', **kwargs)
    transform, err, t = icp.runICP(x0=initialGuess, lb=lowerBounds, ub=upperBounds)

    # Transform data-matrix using ICP result
    Dicp = numpyTransform.transformPoints(transform[-1], D)

    # Plot model points blue and transformed points red
    if False:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.scatter(M[:, 0], M[:, 1], M[:, 2], c='r', marker='o')
        ax.scatter(D[:, 0], D[:, 1], D[:, 2], c='b', marker='^')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.scatter(M[:, 0], M[:, 1], M[:, 2], c='r', marker='o')
        ax.scatter(Dicp[:, 0], Dicp[:, 1], Dicp[:, 2], c='b', marker='^')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax = fig.add_subplot(2, 2, 3)
        ax.plot(t, err, 'x--')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

        plt.show()
    else:
        import visvis as vv
        app = vv.use()
        vv.figure()
        vv.subplot(2, 2, 1)
        vv.plot(M[:, 0], M[:, 1], M[:, 2], lc='b', ls='', ms='o')
        vv.plot(D[:, 0], D[:, 1], D[:, 2], lc='r', ls='', ms='x')
        vv.xlabel('[0,0,1] axis')
        vv.ylabel('[0,1,0] axis')
        vv.zlabel('[1,0,0] axis')
        vv.title('Red: z=sin(x)*cos(y), blue: transformed point cloud')

        # Plot the results
        vv.subplot(2, 2, 2)
        vv.plot(M[:, 0], M[:, 1], M[:, 2], lc='b', ls='', ms='o')
        vv.plot(Dicp[:, 0], Dicp[:, 1], Dicp[:, 2], lc='r', ls='', ms='x')
        vv.xlabel('[0,0,1] axis')
        vv.ylabel('[0,1,0] axis')
        vv.zlabel('[1,0,0] axis')
        vv.title('ICP result')

        # Plot RMS curve
        vv.subplot(2, 2, 3)
        vv.plot(t, err, ls='--', ms='x')
        vv.xlabel('time [s]')
        vv.ylabel('d_{RMS}')
        vv.title('KD-Tree matching')
        if 'optAlg' in kwargs:
            opt2 = nlopt.opt(kwargs['optAlg'], 2)
            vv.title(opt2.get_algorithm_name())
            del opt2
        else:
            vv.title('KD-Tree matching')
        app.Run()

if __name__ == '__main__':
    demo()
#    demo2()
