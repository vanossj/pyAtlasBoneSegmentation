'''
Created on Jan 4, 2012

@author: Jeff
'''
import numpy, time
from math import pi
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import gaussian_filter
import scipy.ndimage
from TriModel import TriModel
from scipy.io import loadmat, savemat
from scipy.ndimage.measurements import label
import scipy.signal
import numpyTransform
from ICP import ICP
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from polygonise import Polygonise
import STL
import os.path
from numpy.ma.core import absolute


def getLargestVolumeLabel(binaryVol):
    labeledVol, num_features = label(binaryVol)
    volumeSize = numpy.zeros(num_features + 1)
    for i in xrange(1, num_features + 1):
        volumeSize[i] = numpy.sum(binaryVol[labeledVol == i])
    maxVolume = volumeSize[1:].max()
    maxVolumeLabel = numpy.nonzero(volumeSize == maxVolume)
    labeledVol[labeledVol != maxVolumeLabel] = 0
    labeledVol[labeledVol == maxVolumeLabel] = 1
    return labeledVol


def filterSpineData(x, data, headTop):
    dataRaw = data.copy()
    dataDiv1 = numpy.zeros(data.shape)
    dataDiv2 = numpy.zeros(data.shape)
    window = numpy.ones(data.shape[0] / 10)
    window /= len(window)
    # smooth the signal a lot
    dataSmoothed = scipy.signal.convolve(data, window, mode='same')
    dataSmoothed = scipy.signal.convolve(dataSmoothed, window, mode='same')
    dataSmoothed = scipy.signal.convolve(dataSmoothed, window, mode='same')
    dataSmoothed = scipy.signal.convolve(dataSmoothed, window, mode='same')

    dataDiv1[:-1] = dataSmoothed[1:] - dataSmoothed[:-1]
    dataDiv1[-1] = dataDiv1[-2]
    dataDiv2[:-1] = dataDiv1[1:] - dataDiv1[:-1]
    dataDiv2[-1] = dataDiv2[-2]

    # cutting 1/7 of front and back of mouse will remove nose and tail
    # this removes some of the noise that occurs at the nose
    startSlice = data.shape[0] / 7
    endSlice = data.shape[0] - startSlice

    dataSmoothed = dataSmoothed[startSlice:endSlice]
    dataDiv1 = dataDiv1[startSlice:endSlice]
    dataDiv2 = dataDiv2[startSlice:endSlice]
    x = x[startSlice:endSlice]
    dataRaw = dataRaw[startSlice:endSlice]

    # look for the first zero crossing of the second derivative, on an increasing slope
    # this point should be the inflection point between the skull and the spine, this is pretty close to the neck position
    if headTop:
        for i in xrange(len(dataDiv2), 0, -1):
            if dataDiv2[i - 1] > 0 and dataDiv2[i] <= 0:
                break
    else:
        for i in xrange(1, len(dataDiv2)):
            if dataDiv2[i - 1] < 0 and dataDiv2[i] >= 0:
                break
    neckPos = x[i]
    return dataRaw, dataSmoothed, dataDiv1, dataDiv2, neckPos, x


def getSpine(filteredVol, neckLoc, headTop, **kwargs):
    '''
    This function assumes that the posterior vector is [1,0,0], ie headTop==False

    function returns volume of spine data as well as hip joint location
    '''
    neckLoc = numpy.array(neckLoc).squeeze()
    hipLoc = None
    if headTop:  # head is in high indices, flip array and neckLoc
        filteredVol = filteredVol[::-1, :, :]
        neckLoc[0] = filteredVol.shape[0] - neckLoc[0]
    neckLocInt = neckLoc.round().astype(numpy.int)
    filteredVol[neckLocInt[0], neckLocInt[1], neckLocInt[2]] = 3.0
    spineVol = numpy.zeros(filteredVol.shape)
    spineVolFinal = numpy.zeros(spineVol.shape)
    tailVolFinal = numpy.zeros(spineVol.shape)
    numBlobDetachFromSpine = 0
    taillabel = None

    for taSliceIndx in xrange(neckLocInt[0], filteredVol.shape[0]):
        taSlice = filteredVol[taSliceIndx]
        taSlice = scipy.ndimage.binary_fill_holes(taSlice)
        taSlicei, numFeaturesi = label(taSlice)
        taSliceLabeled = taSlice.copy()
        taSlicej = spineVol[taSliceIndx - 1]  # previous slice
        taSlice = numpy.zeros(taSlice.shape)  # taSlice will be rebuild using labels
        if 1 not in numpy.unique(taSlicej) and taSliceIndx != neckLocInt[0] and hipLoc is None:
            print 'Spine label is gone!'
        # first time through find blobs, if more that one, find which blob center of mass is closest to neck location, this is the spine blob, ignore other blobs
        if taSliceIndx == neckLocInt[0]:
            if numFeaturesi > 1:
                CoMs = numpy.array(center_of_mass(taSliceLabeled, taSlicei, numpy.unique(taSlicei)[1:]))
                if CoMs.ndim == 1:
                    raise Exception('There are suppose to be multiple labels but yet there is only one center of mass? something is wrong')
                CoMDist = []
                for CoM in CoMs:
                    CoMDist.append(numpy.sqrt(numpy.sum((CoM - neckLoc[1:]) ** 2)))
                CoMDist = numpy.array(CoMDist)
                spineLabel = numpy.where(CoMDist == CoMDist.min())[0][0] + 1  # label of closest blob
                taSlice[taSlicei == spineLabel] = 1
                spineVol[taSliceIndx] = taSlice
            else:  # one blob in first slice, must be good
                spineVol[taSliceIndx] = taSlicei
            continue  # move on to next slice

        for labeli in numpy.unique(taSlicei):  # look at each label in this slice to see if it corresponds to a label in the previous slice
            if labeli == 0:
                continue
            maski = taSlicei == labeli
            for labelj in numpy.unique(taSlicej):  # look at each label in previous slice
                if labelj == 0:
                    continue
                maskj = taSlicej == labelj
                if numpy.any(numpy.logical_and(maski, maskj)):  # if masks have something in common then they are part of the same thing
                    taSlice[maski] = labelj
                    break

        if hipLoc is None and 1 not in numpy.unique(taSlice):  # we haven't found the hip yet and for some reason there isn't a 1 in our slice, make nearest label to previous slice spine label, the spine label for this slice
            CoMj = numpy.array(center_of_mass(taSlicej, taSlicej, 1))  # previous slice spine location
            CoMs = numpy.array(center_of_mass(taSlicei, taSlicei, numpy.unique(taSlicei)[1:]))
            CoMDist = []
            for CoM in CoMs:
                CoMDist.append(numpy.sqrt(numpy.sum((CoM - CoMj) ** 2)))
            CoMDist = numpy.array(CoMDist)
            spineLabel = numpy.where(CoMDist == CoMDist.min())[0][0] + 1  # label of closest blob
            taSlice[taSlicei == spineLabel] = 1

        for labeli in numpy.unique(taSlicei):  # look for labels which are new and weren't in previous slice
            if labeli == 0:
                continue
            for labelNum in xrange(taSlice.size):  # look for next available label number
                if labelNum not in taSlice:
                    break
            maski = taSlicei == labeli
            if numpy.all(taSlice[maski] == 0):  # no overlap must be unique label
                taSlice[maski] = labelNum

        # check to see if a label merged into the spine label
        # spine label will always be 1
        # hip can't be closer than 15 slices from neck position
        if hipLoc is None and taSliceIndx > neckLocInt[0] + 25:
            spineMask = taSlice == 1
            for labelj in numpy.unique(taSlicej):  # look at each label in previous slice
                if labelj == 0 or labelj == 1:
                    continue
                maskj = taSlicej == labelj
                if numpy.any(numpy.logical_and(spineMask, maskj)):  # if masks have something in common then they are part of the same thing, blob merge has been found
                    CoM = center_of_mass(taSlicej, taSlicej, [1])
                    CoM = numpy.array(CoM).squeeze()
                    hipLoc = numpy.array([taSliceIndx, CoM[0], CoM[1]])
                    spineVolFinal[spineVol == 1] = 1
                    print 'Hip joint start location', hipLoc
                    break

        if hipLoc is not None:  # search for end of hipjoint
            taSliceii = numpy.zeros(taSlice.shape)
            taSliceii[taSlice == 1] = 1
            labelii, numFeaturesii = label(taSliceii)
            if numBlobDetachFromSpine == 0 and numFeaturesii > 2:  # spine has split into at least 3 things, this means both hips are separated in the space of one slice, measure distances from CoM of each blob to the hipLoc, closest blob CoM is the spine/tail blob
                numBlobDetachFromSpine += numFeaturesii - 1
                # Get distance between each blob of
                CoMs = numpy.array(center_of_mass(labelii, labelii, numpy.unique(labelii)[1:]))
                if CoMs.ndim == 1:
                    raise Exception('There are suppose to be multiple labels but yet there is only one center of mass? something is wrong')
                CoMDist = []
                for CoM in CoMs:
                    CoMDist.append(numpy.sqrt(numpy.sum((CoM - hipLoc[1:]) ** 2)))
                CoMDist = numpy.array(CoMDist)
                spineLabel = numpy.where(CoMDist == CoMDist.min())[0][0] + 1  # label of closest blob to hiploc
                for lab in numpy.unique(labelii):
                    if lab == 0 or lab == spineLabel:
                        continue
                    for labelnum in xrange(taSlice.size):  # look for next available label number
                        if labelnum not in taSlice:
                            break
                    taSlice[labelii == lab] = labelnum

            elif numBlobDetachFromSpine == 0 and numFeaturesii > 1:  # only one hip has detached from the spine, use the larger of the two blobs as the spine\tail blob
                numBlobDetachFromSpine += numFeaturesii - 1
                labSize = []
                for lab in numpy.unique(labelii):
                    if lab == 0:
                        continue
                    labSize.append(numpy.sum(labelii == lab))
                labSize = numpy.array(labSize)
                spineLabelIndex = numpy.where(labSize == labSize.max())[0][0] + 1  # index in unique() that gets label with largest area
                spineLabel = numpy.unique(labelii)[spineLabelIndex]
                for lab in numpy.unique(labelii):
                    if lab == 0 or lab == spineLabel:
                        continue
                    for labelnum in xrange(taSlice.size):  # look for next available label number
                        if labelnum not in taSlice:
                            break
                    taSlice[labelii == lab] = labelnum

            elif numBlobDetachFromSpine == 1 and numFeaturesii > 1:  # second hip has detached from the spine, measure the distance from CoM of blobs to hiploc, uses closest blob CoM as spine/tail blob
                numBlobDetachFromSpine += numFeaturesii - 1
                # Get distance between each blob of
                CoMs = numpy.array(center_of_mass(labelii, labelii, numpy.unique(labelii)[1:]))
                if CoMs.ndim == 1:
                    raise Exception('There are suppose to be multiple labels but yet there is only one center of mass? something is wrong')
                CoMDist = []
                for CoM in CoMs:
                    CoMDist.append(numpy.sqrt(numpy.sum((CoM - hipLoc[1:]) ** 2)))
                CoMDist = numpy.array(CoMDist)
                spineLabel = numpy.where(CoMDist == CoMDist.min())[0][0] + 1  # label of closest blob to hiploc
                for lab in numpy.unique(labelii):
                    if lab == 0 or lab == spineLabel:
                        continue
                    for labelnum in xrange(taSlice.size):  # look for next available label number
                        if labelnum not in taSlice:
                            break
                    taSlice[labelii == lab] = labelnum

            if numBlobDetachFromSpine >= 2 and taillabel is None:  # both hips have detached from the spine, mark this point as end of spine joint, modify hiploc,
                # modify hipLoc position to be center of hip joint
                CoM = center_of_mass(taSlice, taSlice, [1])
                CoM = numpy.array(CoM).squeeze()
                print 'Hip joint end location', numpy.array([taSliceIndx, CoM[0], CoM[1]])
                hipLoc = (hipLoc + numpy.array([taSliceIndx, CoM[0], CoM[1]])) / 2.0  # get mean between start and end location

                # overwrite spine label with tail
                taillabel = taSlice.size + 1
                taSlice[taSlice == 1] = taillabel

        spineVol[taSliceIndx] = taSlice

    if hipLoc is None:
        raise Exception('Could not find hip')

    # create tail volume
    tailVolFinal[spineVol == taillabel] = 1

    if headTop:
        print 'accounting for aligned volume being flipped'
        hipLoc[0] = filteredVol.shape[0] - hipLoc[0]  # flip first dimension of hipLoc to compensate for original flipping of filteredVol
        spineVolFinal = spineVolFinal[::-1, :, :]  # flip first dimension of spineVolFinal to compensate for original flipping of filteredVol
        tailVolFinal = tailVolFinal[::-1, :, :]  # flip first dimension of tailVolFinal to compensate for original flipping of filteredVol
    print 'Hip location:', hipLoc

    if 'spineModelShortcut' in kwargs and os.path.exists(kwargs['spineModelShortcut']):
        spineModel = STL.readSTLfile(kwargs['spineModelShortcut'], verbose=False)
    else:
        # create mesh of spine
        spineModel = Polygonise(spineVolFinal, 0.5).isosurface()
        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
            STL.saveSTLFile(spineModel, kwargs['savePrefix'] + 'SpineModel.stl', binary=False)

    if 'tailModelShortcut' in kwargs and os.path.exists(kwargs['tailModelShortcut']):
        tailModel = STL.readSTLfile(kwargs['tailModelShortcut'], verbose=False)
    else:
        # create mesh of tail
        tailModel = Polygonise(tailVolFinal, 0.5).isosurface()
        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
            STL.saveSTLFile(tailModel, kwargs['savePrefix'] + 'TailModel.stl', binary=False)

    if 'visualize' in kwargs and kwargs['visualize']:
        import visvis as vv
        for taillabelDisp in xrange(taSlice.size):  # look for next available label number
            if taillabelDisp not in spineVol:
                break
        spineVol[spineVol == taillabel] = taillabelDisp
        app = vv.use()
        vv.figure()
        vv.volshow2(spineVol)
        vv.title('Spine segmentation')
        vv.xlabel('[0,0,1] axis')
        vv.ylabel('[0,1,0] axis')
        vv.zlabel('[1,0,0] axis')
        app.Run()

    return hipLoc, spineModel, tailModel


def roughAlign(scanData, atlasData, **kwargs):
    '''
    roughtAlign() takes the CT volume, CTmesh, and isolevel and returns the
    rotation matrix, scale factor, and neck joint location

    This rough align function is only designed to work for mice, it might work
    for other rodents, but almost certainly would not work for people.

    Rough Align Steps:
    1.    PCA analysis of Mesh points to figure out the axes
            This should results in the axes being defined in a certain order
            [anteroposterior, dorsoventral, left-right]

    2.    Align CTVolume to the standard [x, y, z] cartesian coordinate system
            This can be done by using the PCA axis as a rotation matrix from
            step 1. The resulting alignedCTVolume dimensions are
            [transverse, coronal, sagittal]. For example alignedCTVolume[:,:,1]
            would be a sagittal plane slice.

    3.    Determine the vector that points to the anterior end
            Since we know that the anteroposterior axis is along the [1,0,0]
            line, the vector, as taken from the center point on the
            anteroposterior axis of the alignedCTVolume, that points to the
            anterior end must be [-1,0,0] or [1,0,0]. Measure the amount of bone
            mass in each half of the alignedCTVolume. Halves are created by
            splitting the alignedCTVolume by the [N/2,:,:] plane, where N is
            length of the alignedCTVolume in the first dimension. Bone mass is
            determined as the number of pixels above the isolevel threshold. The
            half with the most bone mass will be the half with the skull. So if
            the alignedCTVolume[0:N/2] half has more bone mass than the
            alignedCTVolume[N/2:N] half, the vector pointing in the direction of
            the anterior end would be [-1,0,0]

    4.    Determine the vector that points to the dorsal side
            We know that the dorsoventral axis is the [0,1,0] axis of the
            alignedCTVolume. So the vector, as taken from the center point on
            the dorsoventral axis of the alignedCTVolume, must be either [0,1,0]
            or [0,-1,0]. The center of mass (CoM) can can be calculated for each
            binary transverse slice. normal transverse slices are transformed
            into binary by thresholding on the isolevel. The line that connects
            these CoM points closely follows the spine along the midsection of
            the animal. The spine curvature in the midsection of a mouse is
            typically a high arch with its apex pointing towards the dorsal
            side. This high arch can be detected because on average the CoM line
            on the arc side is further away from the median in the second
            dimension that the CoM values on the other side of the median line.
            First flatten the 3D curve onto the [:,:,0] plane. Determine the
            mean coordinate, curveMean. Define the second dimension of curveMean
            value B, i.e. B=curveMean[1]. Determine mean point of all points
            whose second dimension value is greater than curveMean[1], call this
            value A. Determine mean point of all points whose second dimension
            value is less than curveMean[1], call this value C. If A > C than
            the vector that points to the dorsal end is [0,1,0]


    5.     Determine Neck Location
            asdf


    6. Determine Scale factor
            asdf
    '''
    # preprocessing
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    else:
        verbose = False
    if 'visualize' in kwargs:
        visualize = kwargs['visualize']
    else:
        visualize = False
    if 'pcaAxis' in kwargs:
        pcaAxis = kwargs['pcaAxis']
    else:
        pcaAxis = None

    # Step 1.    PCA analysis of Mesh points to figure out the axes.
    if pcaAxis is None:
        t1 = time.time()
        pcaAxis = numpy.matrix(scanData.isosurfaceModel.PCA())
        if verbose:
            print 'Time to perform Principle Component Analysis:', time.time() - t1

    # Step 2.    Align CTvolume to the standard [x, y, z] cartesian coordinate system
    rotMat = numpyTransform.coordinateSystemConversionMatrix(pcaAxis, numpy.identity(3), N=3)
    print 'Alignment Rotation Matrix:'
    print rotMat
    alignedCTVolume, rotMat, offset, affineTrans = alignVolume(scanData.ctFieldPreprocessed, rotMat, verbose=verbose)

    # Step 3.    Determine the vector that points to the anterior end
    # make alignedCTVolume binary based on isolevel threshold
    t1 = time.time()
    zeroIndex = alignedCTVolume < scanData.isolevel
    oneIndex = alignedCTVolume >= scanData.isolevel
    alignedCTVolume[zeroIndex] = 0.0
    alignedCTVolume[oneIndex] = 1.0
    if verbose:
        print 'Time to create binary alignCTVolume:', time.time() - t1

    # Determine is anterior diction based on bone mass and translate it back to PCA axis
    t1 = time.time()
    lowerBoneMass = numpy.sum(alignedCTVolume[0:alignedCTVolume.shape[0] / 2, :, :])
    upperBoneMass = numpy.sum(alignedCTVolume[alignedCTVolume.shape[0] / 2:, :, :])
    if upperBoneMass > lowerBoneMass:
        anteriorVector = numpy.asarray(numpy.matrix([1.0, 0.0, 0.0]) * rotMat.I).squeeze()
        headTop = True
    else:
        anteriorVector = numpy.asarray(numpy.matrix([-1.0, 0.0, 0.0]) * rotMat.I).squeeze()
        headTop = False
    if verbose:
        print 'Larger of the two following numbers indicates the anterior direction'
        print 'Lower bone mass:', lowerBoneMass
        print 'Upper bone mass:', upperBoneMass
        print 'Anterior Vector is (aligned to PCA axes):', anteriorVector
        print 'Time to calculate anterior vector:', time.time() - t1

    # Step 4.    Determine the vector that points to the dorsal side
    # Create 3D curve of center of mass points on transverse slices
    # also calculate the standard deviation of the distance from the center of mass to all points above the isolevel in each transverse slice
    # this will be used to estimate the neck location
    t1 = time.time()
    largestVolLabel = getLargestVolumeLabel(alignedCTVolume)  # largestVolLabel used to be alignedCTVolume, using largestVolLabel has the benefit of removing the forelimbs because they aren't directly attached to the rest of the skeleton
    CoMList = []
    coronalData = []
    for i in xrange(largestVolLabel.shape[0]):
        transverseSlice = largestVolLabel[i, :, :]
        indicies = numpy.array(numpy.where(transverseSlice == 1.0))
        if indicies.shape[1] == 0:
            continue
        CoM = center_of_mass(transverseSlice)
        if numpy.any(numpy.isnan(CoM)):
            continue
        CoMList.append([i, CoM[0], CoM[1]])
        coronalData.append(numpy.std(indicies[1] - CoM[1]))
    CoMList = numpy.array(CoMList)
    coronalData = numpy.array(coronalData)
    if verbose:
        print 'Time to calculate 3D Center of Mass Curve and Stats:', time.time() - t1

    # Pick dorsal vector and transform it to Mesh PCA axis
    t1 = time.time()
    meanVal = CoMList.mean(axis=0)
    meanHigh = numpy.mean(CoMList[CoMList[:, 1] > meanVal[1]], axis=0)
    meanLow = numpy.mean(CoMList[CoMList[:, 1] < meanVal[1]], axis=0)
    if abs((meanHigh - meanVal)[1]) > abs((meanLow - meanVal)[1]):
        dorsalVector = numpy.asarray(numpy.matrix([0.0, 1.0, 0.0]) * rotMat.I).squeeze()
    else:
        dorsalVector = numpy.asarray(numpy.matrix([0.0, -1.0, 0.0]) * rotMat.I).squeeze()
    if verbose:
        print 'Greater of following two values indicates that is the dorsal direction'
        print 'Mean of values above mean:', abs((meanHigh - meanVal)[1])
        print 'Mean of values below mean:', abs((meanLow - meanVal)[1])
        print 'Dorsal Vector is (aligned to PCA axes):', dorsalVector
        print 'Time to calculate dorsal vector:', time.time() - t1

    # Step 5: Determine neck location
    # TODO: Document
    t1 = time.time()
    dataRaw, dataSmoothed, dataDiv1, dataDiv2, neckPosAligned, x = filterSpineData(CoMList[:, 0], coronalData, headTop=headTop)
    neckPosAligned = CoMList[CoMList[:, 0] == neckPosAligned][0]
    # transform neck position back to original coordinate system
    neckPosResample = numpyTransform.transformPoints(affineTrans.I, neckPosAligned)

    # scale Neck position by the resampleFactor and slice to appropriate size
    neckPos = neckPosResample * scanData.resampleFactor
    if verbose:
        print 'Neck Position', neckPos
        print 'Time to calculate Neck Position:', time.time() - t1

    # get spine and hip information
    filteredBinVol = numpy.zeros(scanData.ctFieldPreprocessed.shape)
    filteredBinVol[scanData.ctFieldPreprocessed >= scanData.isolevel] = 1.0
    hipLocAligned, spineModel, tailModel = getSpine(largestVolLabel, neckPosAligned, headTop, **kwargs)

    vertices = numpyTransform.transformPoints(affineTrans.I, spineModel.OriginalVertexList) * scanData.resampleFactor
    spineModel = TriModel(vertices, spineModel.TriangleVertexIndexList, name='Spine Model', color=[ 0.83921569, 0. , 0.51764706])

    vertices = numpyTransform.transformPoints(affineTrans.I, tailModel.OriginalVertexList) * scanData.resampleFactor
    tailModel = TriModel(vertices, tailModel.TriangleVertexIndexList, name='Tail Model', color=[ 1, 0, 0])

    # transform the hip position back to non aligned coordinate system
    hipLocResample = numpyTransform.transformPoints(affineTrans.I, hipLocAligned)
    hipLoc = hipLocResample * scanData.resampleFactor
    print 'Hip location dicom', hipLoc

    # Step 6: Determine scaling factor
    # TODO: add documentation
    t1 = time.time()
    atlasSize = atlasData.atlasJoint.getBoundingBox()
    dicomSize = scanData.isosurfaceJoint.getBoundingBox()
    if verbose:
        print 'Atlas Bounding Box:', atlasSize
        print 'DICOM Bounding Box:', dicomSize
    atlasSize = numpy.sqrt(numpy.sum((atlasSize[1] - atlasSize[0]) ** 2))
    dicomSize = numpy.sqrt(numpy.sum((dicomSize[1] - dicomSize[0]) ** 2))
    if verbose:
        print 'Atlas Size:', atlasSize
        print 'DICOM Size:', dicomSize
    scaleFactor = dicomSize / atlasSize
    if verbose:
        print 'Scale Factor:', scaleFactor
        print 'Time to calculate scale factor:', time.time() - t1

    if visualize:
        import visvis as vv  # visvis axis are (z,y,x)
        app = vv.use()

        # for visualization purposes set values so CoM line is visible in volume
        for CoM in CoMList:
            if numpy.all(CoM == neckPosAligned):
                CoM = CoM.round().astype(numpy.int)
                alignedCTVolume[CoM[0], CoM[1], CoM[2]] = 3.0
            else:
                CoM = CoM.round().astype(numpy.int)
                alignedCTVolume[CoM[0], CoM[1], CoM[2]] = 2.0
        # highlight hip joint
        hipindx = hipLocAligned.round().astype(numpy.int)
        alignedCTVolume[hipindx[0], hipindx[1], hipindx[2]] = 3.0

        vv.figure()
        vv.volshow(scanData.ctFieldPreprocessed, renderStyle='mip')
        vv.ColormapEditor(vv.gca())
        vv.title('Start Volume')
        vv.xlabel('[0,0,1] axis')
        vv.ylabel('[0,1,0] axis')
        vv.zlabel('[1,0,0] axis')

        vv.figure()
        vv.volshow(alignedCTVolume, renderStyle='mip')
        vv.ColormapEditor(vv.gca())
        vv.title('Aligned Volume')
        vv.xlabel('[0,0,1] axis')
        vv.ylabel('[0,1,0] axis')
        vv.zlabel('[1,0,0] axis')

        vv.figure()
        vv.plot(CoMList[:, 2], CoMList[:, 1], CoMList[:, 0], lc='k')
        PCAaxis1 = numpy.zeros((3, 3))
        PCAaxis1[2, :] = numpy.asarray(rotMat)[0] * 70
        PCAaxis1 += CoMList.mean(axis=0)
        vv.plot(PCAaxis1[:, 2], PCAaxis1[:, 1], PCAaxis1[:, 0], lc='r')
        PCAaxis2 = numpy.zeros((3, 3))
        PCAaxis2[2, :] = numpy.asarray(rotMat)[1] * 70
        PCAaxis2 += CoMList.mean(axis=0)
        vv.plot(PCAaxis2[:, 2], PCAaxis2[:, 1], PCAaxis2[:, 0], lc='g')
        PCAaxis3 = numpy.zeros((3, 3))
        PCAaxis3[2, :] = numpy.asarray(rotMat)[2] * 70
        PCAaxis3 += CoMList.mean(axis=0)
        vv.plot(PCAaxis3[:, 2], PCAaxis3[:, 1], PCAaxis3[:, 0], lc='b')
        vv.plot([0, dorsalVector[2] * 30], [0, dorsalVector[1] * 30], [0, dorsalVector[0] * 30], lc='c')
        vv.plot([0, anteriorVector[2] * 30], [0, anteriorVector[1] * 30], [0, anteriorVector[0] * 30], lc='m')
        vv.title('Spine Curve')
        vv.xlabel('[0,0,1] axis')
        vv.ylabel('[0,1,0] axis')
        vv.zlabel('[1,0,0] axis')
        vv.gca().SetLimits(rangeX=(0, alignedCTVolume.shape[2]), rangeY=(0, alignedCTVolume.shape[1]), rangeZ=(0, alignedCTVolume.shape[0]))
        vv.gca().legend = 'CoM line aligned to normal axis', 'PCA[0] (anteroposterior) axis', 'PCA[1] axis', 'PCA[2] axis', 'Mouse back direction', 'Mouse head direction'

        vv.figure()
        vv.subplot(4, 1, 1)
        vv.plot(x, dataRaw, lc='k')
        vv.title('Standard Deviation of bone in Coronal Plane')
        vv.subplot(4, 1, 2)
        vv.plot(x, dataSmoothed, lc='r')
        vv.title('Smoothed')
        vv.subplot(4, 1, 3)
        vv.plot(x, dataDiv1, lc='g')
        vv.title('First Derivative')
        vv.subplot(4, 1, 4)
        vv.plot(x, dataDiv2, lc='b')
        vv.title('Second Derivative')

        app.Run()

    # create rotation matrix that will align atlas to CTVolume
    alignmentAxis = numpy.empty((3, 3))
    alignmentAxis[0, :] = anteriorVector
    alignmentAxis[1, :] = dorsalVector
    alignmentAxis[2, :] = numpy.cross(anteriorVector, dorsalVector)

    scanData.alignmentAxis = alignmentAxis


    # Perform PCA of spine model, second vector will be dorsoventral axis
    spinePCA = spineModel.PCA()
    spinePCA[0] = numpy.asarray(numpy.matrix(spinePCA[0]) * rotMat.I).squeeze()
    spinePCA[1] = numpy.asarray(numpy.matrix(spinePCA[1]) * rotMat.I).squeeze()
    spinePCA[2] = numpy.asarray(numpy.matrix(spinePCA[2]) * rotMat.I).squeeze()
#    ang = numpy.arccos(numpy.dot(spinePCA[0], alignmentAxis[0]))
#    rot = numpyTransform.rotation(ang, numpy.cross(spinePCA[0], alignmentAxis[0]), N=4)
#    numpyTransform.transformPoints(rot, ang)

    spinePCA[1] = numpy.cross(alignmentAxis[0], spinePCA[2])
    spinePCA[2] = numpy.cross(alignmentAxis[0], spinePCA[1])

    # determine angles between model PCA and Spine PCA axis, they should be similar (<20 degrees), if not we should to redo the rough align with spine axis
    dif = numpy.zeros(3)
    for i in xrange(spinePCA.shape[0]):
        dif[i] = numpy.min(numpy.degrees(numpy.arccos([numpy.dot(spinePCA[i], alignmentAxis[i]), numpy.dot(-spinePCA[i], alignmentAxis[i])])))
    print 'Anteroposterior axis, spine & model %f degrees apart' % (dif[0])
    print 'Dorsoventral axis, spine & model %f degrees apart' % (dif[1])
    print 'Left-Right axis, spine & model %f degrees apart' % (dif[2])

    return alignmentAxis, neckPos, scaleFactor, hipLoc, spineModel, tailModel, spinePCA


def roughAlignProfile():
    import os, cProfile
    '''
    run
        python -m cProfile -o roughAlign.TimeProfile pyAtlasSegmentation.py
    then run
        python runsnake.py roughAlign.TimeProfile
    '''
    if os.path.exists('roughAlign.TimeProfile'):
        os.remove('roughAlign.TimeProfile')
    cProfile.runctx('RoughAlignTest()', globals(), locals(), filename='roughAlign.TimeProfile')
    os.system('runsnake roughAlign.TimeProfile')


def alignVolume(CTVolume, rotMat, verbose=False):
    # TODO: change function, this is all very confusing because affine_transform() actually does rotMat.I instead of rotMat
    # all point * mat equations should be mat*point equations, they currently work because mat is rotmat, not rotmat.I
    rotMat = numpy.matrix(rotMat)

    # list all vertices defining the bounding box formed by volume
    bbPoints = numpy.array([[0.0, 0.0, 0.0],
                            [0.0, 0.0, CTVolume.shape[2]],
                            [0.0, CTVolume.shape[1], 0.0],
                            [0.0, CTVolume.shape[1], CTVolume.shape[2]],
                            [CTVolume.shape[0], 0.0, 0.0],
                            [CTVolume.shape[0], 0.0, CTVolume.shape[2]],
                            [CTVolume.shape[0], CTVolume.shape[1], 0.0],
                            [CTVolume.shape[0], CTVolume.shape[1], CTVolume.shape[2]]])

    # recalculate limit
    bbPointsRotated = numpy.empty_like(bbPoints)
    for i in xrange(bbPoints.shape[0]):
        bbPointsRotated[i] = numpy.asarray(numpy.matrix(bbPoints[i]) * rotMat).squeeze()
    # parts of volume that are rotated into negative areas
    # Take the minimum values of the rotated bounding box, do the inverse rotation and use the result as the offset to subtract
    offset = bbPointsRotated.min(axis=0)
    offset[offset > 0] = 0
    offset = numpy.matrix(offset) * rotMat.I  # TODO: should this be a transpose instead of invert? does it matter?
    offset = numpy.asarray(offset).squeeze()
    if verbose:
        print 'Offset:', offset
    # calculate shape, for each bounding box vertex subtract offset, then rotate these points, the max coordinate will tell the shape
    points = numpy.empty_like(bbPoints)
    for i in xrange(bbPoints.shape[0]):
        points[i] = numpy.asarray(numpy.matrix(bbPoints[i] - offset) * rotMat).squeeze()
    shape = points.max(axis=0)
#    if verbose:
#        print 'Shape:', shape

    # Transform CTvolume to alignedCTVolume
    if verbose:
        t1 = time.time()

    # affine_transform subtracts offset then rotates
    alignedCTVolume = affine_transform(CTVolume, rotMat, offset=offset, output_shape=shape, cval=-1000)

    # affineTrans is the transform that discribes the transformation done by affine_transform()
    affineTrans = numpy.matrix(numpy.identity(4))
    affineTrans[:3, :3] = rotMat.I
    affineTrans = affineTrans * numpyTransform.translation(offset).I

    if verbose:
        print 'Time to align CT volume: %f seconds' % (time.time() - t1)
        print 'Reference Volume Shape:', CTVolume.shape
        print 'Aligned Volume Shape:  ', alignedCTVolume.shape

    return alignedCTVolume, rotMat, offset, affineTrans


def createLabelMask(scanData, atlasData):
    # TODO: probably don't want to base mask on unprocessed data, probably smooth it, don't need to resample
#    ctVolume = scanData.ctField
    ctVolume = gaussian_filter(scanData.ctField, scanData.sigma)

    labelVolume = numpy.zeros(ctVolume.shape, dtype=numpy.int64)

    t1 = time.time()
    # calculate transformed models
    atlasData.atlasJoint.transformVertices()
    print 'Time to transform all model vertices:', time.time() - t1

    # see what vertex each point in
    # TODO: maybe there is a smarter way of doing this where we first look at the bounding boxes to rule out options
    # faster way, check to see distance to bounding box planes. If these are not better than currentClosestDistance, don't need to check every vertex of this model
    t1 = time.time()
    boneIndices = numpy.where(ctVolume >= scanData.isolevel)
    print 'Number of bone voxels to solve for:', boneIndices[0].shape[0]
    for i in xrange(boneIndices[0].shape[0]):
        t2 = time.time()
        boneIndex = numpy.array([boneIndices[0][i], boneIndices[1][i], boneIndices[2][i]])
        closestSqDistance, modelID, modelName = atlasData.atlasJoint.compareToTransformedPoints(boneIndex)
        labelVolume[boneIndex[0], boneIndex[1], boneIndex[2]] = modelID
#        print '%10.6f%%: Index %s closest to bone ID %d %s. Time to figure this out %f' %(100.0*i/boneIndices[0].shape[0], str(boneIndex), modelID, modelName, time.time()-t2)
    print 'Time to create label volume:', time.time() - t1
    return labelVolume


def createLabelMaskKDTree(scanData, atlasData):
    # TODO: probably don't want to base mask on unprocessed data, probably smooth it, don't need to resample
    # TODO: this is really slow because its a pure python KD tree implementation
#    ctVolume = scanData.ctField
    ctVolume = gaussian_filter(scanData.ctField, scanData.sigma)

    labelVolume = numpy.zeros(ctVolume.shape, dtype=numpy.int64)

    t1 = time.time()
    # calculate transformed models
    atlasData.atlasJoint.transformVertices()
    print 'Time to transform all model vertices:', time.time() - t1

    t1 = time.time()
    boneIndices = numpy.where(ctVolume >= scanData.isolevel)
    print 'Number of bone voxels to solve for:', boneIndices[0].shape[0]
    for i in xrange(boneIndices[0].shape[0]):
        t2 = time.time()
        boneIndex = numpy.array([boneIndices[0][i], boneIndices[1][i], boneIndices[2][i]])
        closestSqDistance, modelID, modelName = atlasData.atlasJoint.compareToTransformedPointsKDTrees(boneIndex)
        labelVolume[boneIndex[0], boneIndex[1], boneIndex[2]] = modelID
#        print '%10.6f%%: Index %s closest to bone ID %d %s. Time to figure this out %f' %(100.0*i/boneIndices[0].shape[0], str(boneIndex), modelID, modelName, time.time()-t2)
    print 'Time to create label volume:', time.time() - t1
    return labelVolume


def RoughAlignTest(**kwargs):
    # Start with saved data
    savedData = loadmat('Saved Data.mat')

#    returnValues  = roughAlign(savedData['referenceVolume'], None, 350, pcaAxis=savedData['axes'],alignedCTVolume=savedData['alignedCTVolume'],**kwargs)
    returnValues = roughAlign(savedData['referenceVolume'], None, 350, pcaAxis=savedData['axes'], **kwargs)
    return returnValues


def FineAlign(scanData, atlasData, **kwargs):
    visualize = True
    # apply transform to CT dataset, 1 time thing, this is to account for scaling
    scanData.isosurfaceJoint.transformVertices()
    iso1KDTree = KDTree(scanData.isosurfaceModel1.transformedVertexList)
    tailKDTree = KDTree(kwargs['tailVertices'])
    spineKDTree = KDTree(kwargs['spineVertices'])

    if 'fineAlignTransformsShortcut' in kwargs:
        fineAlignTransforms = loadmat(kwargs['fineAlignTransformsShortcut'])
    else:
        fineAlignTransforms = {}

    if 'spineVertexShortcut' in kwargs:
        vertexDict = loadmat(kwargs['spineVertexShortcut'])
        spineindx = vertexDict['spineindx'].astype(numpy.bool).squeeze()
        tailindx = vertexDict['tailindx'].astype(numpy.bool).squeeze()
    else:
        # find spine indices
        t1 = time.time()
        spineindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
        spineNN = spineKDTree.query_ball_tree(iso1KDTree, 1.0 / scanData.sliceThickness)
        for spinevindx in spineNN:
            spineindx[spinevindx] = True
        print 'Took %f seconds to calculate spine vertices' % (time.time() - t1)

        # find tail indices
        t1 = time.time()
        tailindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
        tailNN = tailKDTree.query_ball_tree(iso1KDTree, 1.0 / scanData.sliceThickness)
        for tailvindx in tailNN:
            tailindx[tailvindx] = True
        print 'Took %f seconds to calculate tail vertices' % (time.time() - t1)
        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
            savemat(kwargs['savePrefix'] + 'spineIndices.mat', {'spineindx': spineindx, 'tailindx': tailindx})

    ############################################################################################################################################
    # Align Skull
    jointOfInterest = atlasData.atlasJoint
    cummulativeJointTransform = atlasData.atlasJoint.getCummulativeTransform(id(jointOfInterest))
    bones = ['Skull Outside', 'Skull Inside']
    if 'Atlas Base' not in fineAlignTransforms:  # Do ICP
        # step 1: get data vertex list. These vertexes are based on the current best transform
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))

        # step 2: Filter model points as much as possible, ideally only bones to be matched will remain
#        modelIndx = numpy.ones(scanData.isosurfaceModel.transformedVertexList.shape[0], dtype=numpy.bool)
        modelIndx = numpy.ones(scanData.isosurfaceModel1.transformedVertexList.shape[0], dtype=numpy.bool)

#        #remove model data that is behind the neck
#        normV = scanData.alignmentAxis[0]
#        planePoint = jointOfInterest.location
#        pointVecs = scanData.isosurfaceModel.transformedVertexList - planePoint
#        distance = numpy.dot(pointVecs, normV)
#        modelIndx[distance < 0] = False

        # create filtered model point cloud
#        modelPoints = scanData.isosurfaceModel.transformedVertexList.copy()
        modelPoints = scanData.isosurfaceModel1.transformedVertexList.copy()

        # Step 3: Reorient Point Clouds
        # transform points back so that joint of interest is in original position/orientation
        # the reason for this is that P=C*V then P = M*P, is not equal to P = (C*M)*V. Where C is parent transform, M is new ICP transform, V is original vector
        # ICP must be done on original points not points midway through a transformation
        modelPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, modelPoints)
        dataPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, dataPoints)

        # move point clouds so that joint is at origin
        jointLocation = jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
        modelPoints -= jointLocation
        dataPoints -= jointLocation

        # limit points to only those reachable given the joint DOF
        mindx = numpyTransform.pointsInToleranceRange(modelPoints, jointOfInterest.DOFvec, jointOfInterest.DOFangle, [jointOfInterest.DOFtrans, jointOfInterest.DOFtrans, jointOfInterest.DOFtrans])
        modelPoints = modelPoints[numpy.logical_and(mindx, modelIndx)]

        # TODO: Convert the model and data points so that the DOFvec is aligned with the z? axis this might allow better control of angles (rotz * rotx * rotz)

        # Step 4: Perform actual ICP
        icp = ICP(modelPoints, dataPoints, maxIterations=15, modelDownsampleFactor=1, dataDownsampleFactor=1, minimizeMethod='fmincon')
        # +- 10mm translation
#        transBound = jointOfInterest.DOFtrans / (scanData.resampleFactor * scanData.sliceThickness)
        transBound = jointOfInterest.DOFtrans / scanData.sliceThickness
        print '%s translation limited to %fmm aka %f pixels' % (jointOfInterest.name, jointOfInterest.DOFtrans, transBound)
        initialGuess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        lowerBounds = numpy.array([-pi, -pi, -pi, -transBound, -transBound, -transBound, 0.8, 0.8, 0.8])
        upperBounds = numpy.array([pi, pi, pi, transBound, transBound, transBound, 1.2, 1.2, 1.2])
        # run ICP
        transform, err, t = icp.runICP(x0=initialGuess, lb=lowerBounds, ub=upperBounds)
        transform = transform[-1]
        print 'ICP Generated Transform for %s Joint' % (jointOfInterest.name)
        print transform

        if visualize:  # display err plot of ICP
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(t, err, 'x--')
            ax.set_xlabel('Time')
            ax.set_ylabel('RMS error')
            ax.set_title('Result of ICP on %s Joint' % (jointOfInterest.name))
            plt.show()

        if visualize and False:
            # entire model
            allModelPoints = scanData.isosurfaceModel1.transformedVertexList.copy()
            allModelPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, allModelPoints)
            allModelPoints -= jointLocation
            tri = numpy.array(range(allModelPoints.shape[0] - allModelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(allModelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

            # Display initial filtered points clouds that will be passed to ICP
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed = numpyTransform.transformPoints(transform, dataPoints)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

            # display point clouds at final position
            modelPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            modelPoints = numpyTransform.transformPoints(cummulativeJointTransform, modelPoints)
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            dataPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPoints = numpyTransform.transformPoints(cummulativeJointTransform, dataPoints)
            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPointsTransformed = numpyTransform.transformPoints(cummulativeJointTransform, dataPointsTransformed)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Final Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

    else:  # skip ICP, just use predetermined transform
        transform = numpy.matrix(fineAlignTransforms['Atlas Base'])
#        transform = numpy.matrix(
#            [[  9.99422678e-01,   1.63207412e-03,   3.39359326e-02,  -8.70562491e-01],
#             [ -2.48435746e-03,   9.99682172e-01,   2.50875024e-02,  -8.82143072e+00],
#             [ -3.38842021e-02,  -2.51573278e-02,   9.99109088e-01,   1.04611786e+01],
#             [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    # store transform
    fineAlignTransforms[jointOfInterest.name] = transform

    # Step 5: Apply ICP transform to joint
    R = numpy.matrix(numpy.identity(4))
    R[:3, :3] = transform[:3, :3]
    jointOfInterest.rotate(R, relative=True)
    cummulativeJointRotation = numpy.matrix(numpy.identity(4))
    cummulativeJointRotation[:3, :3] = cummulativeJointTransform[:3, :3]
    modifiedTranslate = cummulativeJointRotation * numpyTransform.translation(transform[:3, 3].getA().squeeze())  # translation has to include rotation effects of cumulative transformations
    jointOfInterest.translate(modifiedTranslate[:3, 3].getA().squeeze(), absolute=False)

    if 'updateSceneFunc' in kwargs and kwargs['updateSceneFunc'] is not None:
        kwargs['updateSceneFunc']()
        time.sleep(3)

    # print resulting joint location
    if 'getResults' in kwargs and kwargs['getResults']:
        atlasData.atlasJoint.transformVertices()
        print '%s Joint Location: %s' % (jointOfInterest.name, str(jointOfInterest.location))
        meanDist = []
        for bone in bones:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            meanDist.append(numpy.sum(d) / len(i))
        print '%s Joint Mean surface to bone distance: %f' % (jointOfInterest.name, numpy.mean(meanDist))
        dist = 0.0
        numPoints = 0.0
        for bone in scanData.bonesVol1:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            dist += numpy.sum(d)
            numPoints += len(i)
        print 'Post %s Align Mean distance all bones to surface: %f' % (jointOfInterest.name, dist / numPoints)

    if False:  # getting the skull indices might not be necessary if we are blindly removing front half anyway.
        # get indices of model vertices within 1mm of aligned atlas skull vertices
        if 'skullVertexShortcut' in kwargs and os.path.exists(kwargs['skullVertexShortcut']):
            vertexDict = loadmat(kwargs['skullVertexShortcut'])
            skullindx = vertexDict['indx'].astype(numpy.bool).squeeze()
        else:
            # create KDTree of newly aligned bone
            dataPoints = []
            for bone in bones:
                atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
                dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
            dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))
            boneKDTree = KDTree(dataPoints)
            # find spine indices
            t1 = time.time()
            skullindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
            NN = boneKDTree.query_ball_tree(iso1KDTree, 1.0 / scanData.sliceThickness)
            for vindx in NN:
                skullindx[vindx] = True
            print 'Took %f seconds to calculate %s vertices' % (time.time() - t1, bone)
            if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
                savemat(kwargs['savePrefix'] + 'skullvertices.mat', {'indx': skullindx})
    else:
        skullindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)

    ############################################################################################################################################
    # align neck and all subjoints
    for joint in atlasData.atlasJoint.childJoints:
        if joint.name == 'Neck':
            neckJoint = joint
            break
    else:
        raise Exception('Correct Joint not found')

    ############################################################################################################################################
    # translate hip complex to the previously determined hip location
    for joint in neckJoint.childJoints:
        if joint.name == 'Hip Complex':
            hipComplexJoint = joint
            break
    else:
        raise Exception('Correct Joint not found')
    if 'hipLocation' in kwargs:  # hip joint needs to be defined already
        jointOfInterest = hipComplexJoint
        cummulativeParentJointTransform = atlasData.atlasJoint.getCummulativeTransform(id(jointOfInterest.parentJoint))
        atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels['Pelvis Right']))
        hipStartLocation = atlasData.atlasModels['Pelvis Right'].joint.parentJoint.location.squeeze()
        hipLocation = kwargs['hipLocation']
        hipoffset = hipLocation - hipStartLocation
        # need to compensate for the effects of parent joint cumulative transform
        tform = numpy.matrix(numpy.identity(4))
        tform[:3, :3] = cummulativeParentJointTransform[:3, :3]
        hipoffset = numpyTransform.transformPoints(tform.I, hipoffset)
        jointOfInterest.translate(hipoffset, absolute=False)
        atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels['Pelvis Right']))

    ############################################################################################################################################
    # align Left Hip
    for joint in hipComplexJoint.childJoints:
        if joint.name == 'Hip Left':
            hipLeft = joint
            break
    else:
        raise Exception('Correct Joint not found')
    jointOfInterest = hipLeft
    cummulativeJointTransform = atlasData.atlasJoint.getCummulativeTransform(id(jointOfInterest))
    bones = ['Pelvis Left']
    if 'Hip Left' not in fineAlignTransforms:  # Do ICP
        # step 1: get data vertex list. These vertexes are based on the current best transform
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))

        # step 2: Filter model points as much as possible, ideally only bones to be matched will remain
        # create filtered model point cloud
        modelPoints = scanData.isosurfaceModel1.transformedVertexList.copy()
        modelIndx = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # remove front half of model data
        normV = scanData.alignmentAxis[0]
        planePoint = (scanData.isosurfaceModel.maxPointTransformed + scanData.isosurfaceModel.minPointTransformed) / 2.0
        pointVecs = modelPoints - planePoint
        distance = numpy.dot(pointVecs, normV)
        modelIndx[distance > 0] = False

#        #remove right half of model data
#        normV = scanData.alignmentAxis[2]
#        planePoint = jointOfInterest.location
#        pointVecs = modelPoints - planePoint
#        distance = numpy.dot(pointVecs, normV)
#        modelIndx[distance > 0] = False

        # remove previously aligned vertecies
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(spineindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tailindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(skullindx))

        # Step 3: Reorient Point Clouds
        # transform points back so that joint of interest is in original position/orientation
        # the reason for this is that P=C*V then P = M*P, is not equal to P = (C*M)*V. Where C is parent transform, M is new ICP transform, V is original vector
        # ICP must be done on original points not points midway through a transformation
        modelPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, modelPoints)
        dataPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, dataPoints)

        # move point clouds so that joint is at origin
        jointLocation = jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
        modelPoints -= jointLocation
        dataPoints -= jointLocation

        # Remove any points now that model is reoriented
        modelIndx2 = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # Get just points of interest
        modelPoints = modelPoints[numpy.logical_and(modelIndx, modelIndx2)]

        # limit translation
#        transBound = jointOfInterest.DOFtrans / (scanData.resampleFactor * scanData.sliceThickness)
        transBound = jointOfInterest.DOFtrans / scanData.sliceThickness
        print '%s translation limited to %fmm aka %f pixels' % (jointOfInterest.name, jointOfInterest.DOFtrans, transBound)
        # Step 4: Perform actual ICP
        icp = ICP(modelPoints, dataPoints, maxIterations=15, modelDownsampleFactor=1, dataDownsampleFactor=1, minimizeMethod='fmincon')
        initialGuess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        lowerBounds = numpy.array([-pi, -pi, -pi, -transBound, -transBound, -transBound, 0.8, 0.8, 0.8])
        upperBounds = numpy.array([pi, pi, pi, transBound, transBound, transBound, 1.2, 1.2, 1.2])
        transform, err, t = icp.runICP(x0=initialGuess, lb=lowerBounds, ub=upperBounds)
        transform = transform[-1]
        print 'ICP Generated Transform for %s Joint' % (jointOfInterest.name)
        print transform

        if visualize:  # display err plot of ICP
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(t, err, 'x--')
            ax.set_xlabel('Time')
            ax.set_ylabel('RMS error')
            ax.set_title('Result of ICP on %s Joint' % (jointOfInterest.name))
            plt.show()

        if visualize and False:
            # Display initial filtered points clouds that will be passed to ICP
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed = numpyTransform.transformPoints(transform, dataPoints)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

            # display point clouds at final position
            modelPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            modelPoints = numpyTransform.transformPoints(cummulativeJointTransform, modelPoints)
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            dataPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPoints = numpyTransform.transformPoints(cummulativeJointTransform, dataPoints)
            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPointsTransformed = numpyTransform.transformPoints(cummulativeJointTransform, dataPointsTransformed)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Final Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

    else:  # skip ICP, just use predetermined transform
        transform = numpy.matrix(fineAlignTransforms['Hip Left'])
#        transform = numpy.matrix(
#            [[ 0.94610378, -0.28824402, -0.14765847, -2.10207738],
#             [ 0.19292585,  0.86781307, -0.45790839, -3.71208068],
#             [ 0.2601293,   0.40474173,  0.87665095,  7.22883351],
#             [ 0.,          0.,          0.,          1.        ]])

    # store transform
    fineAlignTransforms[jointOfInterest.name] = transform

    # Step 5: Apply ICP transform to joint
    R = numpy.matrix(numpy.identity(4))
    R[:3, :3] = transform[:3, :3]
    jointOfInterest.rotate(R, relative=True)
    cummulativeJointRotation = numpy.matrix(numpy.identity(4))
    cummulativeJointRotation[:3, :3] = cummulativeJointTransform[:3, :3]
    modifiedTranslate = cummulativeJointRotation * numpyTransform.translation(transform[:3, 3].getA().squeeze())  # translation has to include rotation effects of cumulative transformations
#    jointOfInterest.translate(modifiedTranslate[:3,3].getA().squeeze(), absolute=False)
    jointOfInterest.translate(transform[:3, 3].getA().squeeze(), absolute=False)

    if 'updateSceneFunc' in kwargs and kwargs['updateSceneFunc'] is not None:
        kwargs['updateSceneFunc']()
        time.sleep(3)

    # print resulting joint location
    if 'getResults' in kwargs and kwargs['getResults']:
        atlasData.atlasJoint.transformVertices()
        print '%s Joint Location: %s' % (jointOfInterest.name, str(jointOfInterest.location))
        meanDist = []
        for bone in bones:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            meanDist.append(numpy.sum(d) / len(i))
        print '%s Joint Mean surface to bone distance: %f' % (jointOfInterest.name, numpy.mean(meanDist))
        dist = 0.0
        numPoints = 0.0
        for bone in scanData.bonesVol1:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            dist += numpy.sum(d)
            numPoints += len(i)
        print 'Post %s Align Mean distance all bones to surface: %f' % (jointOfInterest.name, dist / numPoints)

    # get indices of model vertices within 1mm of aligned atlas hip left vertices
    if 'hipLeftVertexShortcut' in kwargs and os.path.exists(kwargs['hipLeftVertexShortcut']):
        vertexDict = loadmat(kwargs['hipLeftVertexShortcut'])
        hipLeftindx = vertexDict['indx'].astype(numpy.bool).squeeze()
    else:
        # create KDTree of newly aligned bone
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))
        boneKDTree = KDTree(dataPoints)
        # find spine indices
        t1 = time.time()
        hipLeftindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
        NN = boneKDTree.query_ball_tree(iso1KDTree, 1.0 / scanData.sliceThickness)
        for vindx in NN:
            hipLeftindx[vindx] = True
        print 'Took %f seconds to calculate %s vertices' % (time.time() - t1, bone)
        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
            savemat(kwargs['savePrefix'] + 'hipLeftvertices.mat', {'indx': hipLeftindx})

    ############################################################################################################################################
    # align Right Hip
    for joint in hipComplexJoint.childJoints:
        if joint.name == 'Hip Right':
            hipRight = joint
            break
    else:
        raise Exception('Correct Joint not found')
    jointOfInterest = hipRight
    cummulativeJointTransform = atlasData.atlasJoint.getCummulativeTransform(id(jointOfInterest))
    bones = ['Pelvis Right']
    if 'Hip Right' not in fineAlignTransforms:  # Do ICP
        # step 1: get data vertex list. These vertexes are based on the current best transform
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))

        # step 2: Filter model points as much as possible, ideally only bones to be matched will remain
        # create filtered model point cloud
        modelPoints = scanData.isosurfaceModel1.transformedVertexList.copy()
        modelIndx = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # remove front half of model data
        normV = scanData.alignmentAxis[0]
        planePoint = (scanData.isosurfaceModel.maxPointTransformed + scanData.isosurfaceModel.minPointTransformed) / 2.0
        pointVecs = modelPoints - planePoint
        distance = numpy.dot(pointVecs, normV)
        modelIndx[distance > 0] = False

#        #remove Left half of model data
#        normV = scanData.alignmentAxis[2]
#        planePoint = jointOfInterest.location
#        pointVecs = modelPoints - planePoint
#        distance = numpy.dot(pointVecs, normV)
#        modelIndx[distance < 0] = False

        # remove previously aligned vertices
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(spineindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tailindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(skullindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipLeftindx))

        # Step 3: Reorient Point Clouds
        # transform points back so that joint of interest is in original position/orientation
        # the reason for this is that P=C*V then P = M*P, is not equal to P = (C*M)*V. Where C is parent transform, M is new ICP transform, V is original vector
        # ICP must be done on original points not points midway through a transformation
        modelPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, modelPoints)
        dataPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, dataPoints)

        # move point clouds so that joint is at origin
        jointLocation = jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
        modelPoints -= jointLocation
        dataPoints -= jointLocation

        # Remove any points now that model is reoriented
        modelIndx2 = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # Get just points of interest
        modelPoints = modelPoints[numpy.logical_and(modelIndx, modelIndx2)]

        # limit translation
#        transBound = jointOfInterest.DOFtrans / (scanData.resampleFactor * scanData.sliceThickness)
        transBound = jointOfInterest.DOFtrans / scanData.sliceThickness
        print '%s translation limited to %fmm aka %f pixels' % (jointOfInterest.name, jointOfInterest.DOFtrans, transBound)
        # Step 4: Perform actual ICP
        icp = ICP(modelPoints, dataPoints, maxIterations=15, modelDownsampleFactor=1, dataDownsampleFactor=1, minimizeMethod='fmincon')
        initialGuess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        lowerBounds = numpy.array([-pi, -pi, -pi, -transBound, -transBound, -transBound, 0.8, 0.8, 0.8])
        upperBounds = numpy.array([pi, pi, pi, transBound, transBound, transBound, 1.2, 1.2, 1.2])
        transform, err, t = icp.runICP(x0=initialGuess, lb=lowerBounds, ub=upperBounds)
        transform = transform[-1]
        print 'ICP Generated Transform for %s Joint' % (jointOfInterest.name)
        print transform

        if visualize:  # display err plot of ICP
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(t, err, 'x--')
            ax.set_xlabel('Time')
            ax.set_ylabel('RMS error')
            ax.set_title('Result of ICP on %s Joint' % (jointOfInterest.name))
            plt.show()

        if visualize and False:
            # Display initial filtered points clouds that will be passed to ICP
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed = numpyTransform.transformPoints(transform, dataPoints)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

            # display point clouds at final position
            modelPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            modelPoints = numpyTransform.transformPoints(cummulativeJointTransform, modelPoints)
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            dataPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPoints = numpyTransform.transformPoints(cummulativeJointTransform, dataPoints)
            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPointsTransformed = numpyTransform.transformPoints(cummulativeJointTransform, dataPointsTransformed)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Final Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

    else:  # skip ICP, just use predetermined transform
        transform = numpy.matrix(fineAlignTransforms['Hip Right'])
#        transform = numpy.matrix(
#            [[ 0.96487629, -0.07852993, -0.25069264,  5.96225278],
#             [-0.03107695,  0.91345488, -0.40575166,  4.20753782],
#             [ 0.26086007,  0.39929092,  0.87893048,  4.29578447],
#             [ 0.,          0.,          0.,          1.        ]])

    # store transform
    fineAlignTransforms[jointOfInterest.name] = transform

    # Step 5: Apply ICP transform to joint
    R = numpy.matrix(numpy.identity(4))
    R[:3, :3] = transform[:3, :3]
    jointOfInterest.rotate(R, relative=True)
    cummulativeJointRotation = numpy.matrix(numpy.identity(4))
    cummulativeJointRotation[:3, :3] = cummulativeJointTransform[:3, :3]
    modifiedTranslate = cummulativeJointRotation * numpyTransform.translation(transform[:3, 3].getA().squeeze())  # translation has to include rotation effects of cumulative transformations
#    jointOfInterest.translate(modifiedTranslate[:3,3].getA().squeeze(), absolute=False)
    jointOfInterest.translate(transform[:3, 3].getA().squeeze(), absolute=False)

    if 'updateSceneFunc' in kwargs and kwargs['updateSceneFunc'] is not None:
        kwargs['updateSceneFunc']()
        time.sleep(3)

    # print resulting joint location
    if 'getResults' in kwargs and kwargs['getResults']:
        atlasData.atlasJoint.transformVertices()
        print '%s Joint Location: %s' % (jointOfInterest.name, str(jointOfInterest.location))
        meanDist = []
        for bone in bones:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            meanDist.append(numpy.sum(d) / len(i))
        print '%s Joint Mean surface to bone distance: %f' % (jointOfInterest.name, numpy.mean(meanDist))
        dist = 0.0
        numPoints = 0.0
        for bone in scanData.bonesVol1:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            dist += numpy.sum(d)
            numPoints += len(i)
        print 'Post %s Align Mean distance all bones to surface: %f' % (jointOfInterest.name, dist / numPoints)


    # get indices of model vertices within 1mm of aligned atlas hip left vertices
    if 'hipRightVertexShortcut' in kwargs and os.path.exists(kwargs['hipRightVertexShortcut']):
        vertexDict = loadmat(kwargs['hipRightVertexShortcut'])
        hipRightindx = vertexDict['indx'].astype(numpy.bool).squeeze()
    else:
        # create KDTree of newly aligned bone
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))
        boneKDTree = KDTree(dataPoints)
        # find spine indices
        t1 = time.time()
        hipRightindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
        NN = boneKDTree.query_ball_tree(iso1KDTree, 1.0 / scanData.sliceThickness)
        for vindx in NN:
            hipRightindx[vindx] = True
        print 'Took %f seconds to calculate %s vertices' % (time.time() - t1, bone)
        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
            savemat(kwargs['savePrefix'] + 'hipRightvertices.mat', {'indx': hipRightindx})

    ############################################################################################################################################
    # align Left Femur
    for joint in hipLeft.childJoints:
        if joint.name == 'Pelvis Hindlimb Left':
            femurLeft = joint
            break
    else:
        raise Exception('Correct Joint not found')
    jointOfInterest = femurLeft
    cummulativeJointTransform = atlasData.atlasJoint.getCummulativeTransform(id(jointOfInterest))
    bones = ['Upper Hindlimb Left']
    if 'Pelvis Hindlimb Left' not in fineAlignTransforms:  # Do ICP
#    if True:    #Do ICP
        # step 1: get data vertex list. These vertexes are based on the current best transform
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))

        # step 2: Filter model points as much as possible, ideally only bones to be matched will remain
        # create filtered model point cloud
        modelPoints = scanData.isosurfaceModel1.transformedVertexList.copy()
        modelIndx = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # remove front half of model data
        normV = scanData.alignmentAxis[0]
        planePoint = (scanData.isosurfaceModel.maxPointTransformed + scanData.isosurfaceModel.minPointTransformed) / 2.0
        pointVecs = modelPoints - planePoint
        distance = numpy.dot(pointVecs, normV)
        modelIndx[distance > 0] = False

        # remove previously aligned vertecies
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(spineindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tailindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(skullindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipRightindx))

        # Step 3: Reorient Point Clouds
        # transform points back so that joint of interest is in original position/orientation
        # the reason for this is that P=C*V then P = M*P, is not equal to P = (C*M)*V. Where C is parent transform, M is new ICP transform, V is original vector
        # ICP must be done on original points not points midway through a transformation
        modelPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, modelPoints)
        dataPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, dataPoints)

        # move point clouds so that joint is at origin
        jointLocation = jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
        modelPoints -= jointLocation
        dataPoints -= jointLocation

        # Remove any points now that model is reoriented
        modelIndx2 = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # Get just points of interest
        modelPoints = modelPoints[numpy.logical_and(modelIndx, modelIndx2)]

        # create different starting transformations
        cummJointTransferRotOnly = numpy.matrix(numpy.identity(4))
        cummJointTransferRotOnly[:3, :3] = cummulativeJointTransform[:3, :3]
        proximodistalVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.proximodistalVecTransformed)
        secondaryVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.secondaryVecTransformed)
        initialTransforms = []
        initialTransformsError = []
        icpTransformsErr = []
        icpTransforms = []
        icpT = []

        for spin in numpy.arange(0, 2 * numpy.pi, 3 * numpy.pi):
            for elevation in numpy.arange(0, numpy.pi, numpy.pi / 2):
                for aximuth in numpy.arange(0, 2 * numpy.pi, numpy.pi / 2):
                    initialTransforms.append(numpyTransform.rotation(aximuth, proximodistalVecTransformed, N=4) * numpyTransform.rotation(elevation, secondaryVecTransformed, N=4) * numpyTransform.rotation(spin, proximodistalVecTransformed, N=4))

        for i in xrange(len(initialTransforms)):
            initialTransform = initialTransforms[i]
            dataPointsInitialTrandform = numpyTransform.transformPoints(initialTransform, dataPoints)

            # limit translation
#            transBound = jointOfInterest.DOFtrans / (scanData.resampleFactor * scanData.sliceThickness)
            transBound = jointOfInterest.DOFtrans / scanData.sliceThickness
            print '%s translation limited to %fmm aka %f pixels' % (jointOfInterest.name, jointOfInterest.DOFtrans, transBound)
            # Step 4: Perform actual ICP
            print 'ITERATION', i
            icp = ICP(modelPoints, dataPointsInitialTrandform, maxIterations=15, modelDownsampleFactor=1, dataDownsampleFactor=1, minimizeMethod='fmincon')
            initialGuess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            lowerBounds = numpy.array([-pi, -pi, -pi, -transBound, -transBound, -transBound, 0.8, 0.8, 0.8])
            upperBounds = numpy.array([pi, pi, pi, transBound, transBound, transBound, 1.2, 1.2, 1.2])
            transform, err, t = icp.runICP(x0=initialGuess, lb=lowerBounds, ub=upperBounds)
            del icp
            transform = transform[-1]
            print 'ICP Generated Transform for %s Joint' % (jointOfInterest.name)
            print transform

            icpT.append(t)
            icpTransformsErr.append(err)
            icpTransforms.append(transform)

            if visualize and False:  # display err plot of ICP
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(t, err, 'x--')
                ax.set_xlabel('Time')
                ax.set_ylabel('RMS error')
                ax.set_title('Result of ICP on %s Joint, initial transform %d' % (jointOfInterest.name, len(initialTransformsError)))
                plt.show()

        icpTransformsErr = numpy.array(icpTransformsErr)

        tindx = numpy.where(icpTransformsErr == icpTransformsErr.min())[0][0]
        transform = icpTransforms[tindx] * initialTransforms[tindx]
        print 'Best iteration was %d with transform:' % (tindx)
        print transform

        if visualize:  # display err plot of ICP
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(icpT[tindx], icpTransformsErr[tindx], 'x--')
            ax.set_xlabel('Time')
            ax.set_ylabel('RMS error')
            ax.set_title('Result of Scale ICP on %s Joint, initial transform %d' % (jointOfInterest.name, tindx))
            plt.show()

        if visualize and False:
            # Display initial filtered points clouds that will be passed to ICP
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed = numpyTransform.transformPoints(transform, dataPoints)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

            # display point clouds at final position
            modelPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            modelPoints = numpyTransform.transformPoints(cummulativeJointTransform, modelPoints)
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            dataPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPoints = numpyTransform.transformPoints(cummulativeJointTransform, dataPoints)
            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPointsTransformed = numpyTransform.transformPoints(cummulativeJointTransform, dataPointsTransformed)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Final Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

    else:  # skip ICP, just use predetermined transform
        transform = numpy.matrix(fineAlignTransforms['Pelvis Hindlimb Left'])

    # store transform
    fineAlignTransforms[jointOfInterest.name] = transform

    # Step 5: Apply ICP transform to joint
    R = numpy.matrix(numpy.identity(4))
    R[:3, :3] = transform[:3, :3]
    jointOfInterest.rotate(R, relative=True)
    cummulativeJointRotation = numpy.matrix(numpy.identity(4))
    cummulativeJointRotation[:3, :3] = cummulativeJointTransform[:3, :3]
    modifiedTranslate = cummulativeJointRotation * numpyTransform.translation(transform[:3, 3].getA().squeeze())  # translation has to include rotation effects of cumulative transformations
#    jointOfInterest.translate(modifiedTranslate[:3,3].getA().squeeze(), absolute=False)
    jointOfInterest.translate(transform[:3, 3].getA().squeeze(), absolute=False)

#    if 'updateSceneFunc' in kwargs and kwargs['updateSceneFunc'] is not None:
#        kwargs['updateSceneFunc']()
#        time.sleep(3)
#
#    #print resulting joint location
#    if 'getResults' in kwargs and kwargs['getResults']:
#        atlasData.atlasJoint.transformVertices()
#        print '%s Joint Location: %s' % (jointOfInterest.name, str(jointOfInterest.location))
#        meanDist = []
#        for bone in bones:
#            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
#            meanDist.append(numpy.sum(d)/len(i))
#        print '%s Joint Mean surface to bone distance: %f' % (jointOfInterest.name, numpy.mean(meanDist))
#        dist = 0.0
#        numPoints = 0.0
#        for bone in scanData.bonesVol1:
#            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
#            dist += numpy.sum(d)
#            numPoints += len(i)
#        print 'Post %s Align Mean distance all bones to surface: %f' % (jointOfInterest.name, dist/numPoints)
#
#    #get indices of model vertices within 1mm of aligned atlas hip left vertices
#    if 'femurLeftVertexShortcut' in kwargs and os.path.exists(kwargs['femurLeftVertexShortcut']):
#        vertexDict = loadmat(kwargs['femurLeftVertexShortcut'])
#        femurLeftindx = vertexDict['indx'].astype(numpy.bool).squeeze()
#    else:
#        #create KDTree of newly aligned bone
#        dataPoints = []
#        for bone in bones:
#            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
#            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
#        dataPoints = dataPoints.reshape( (dataPoints.shape[0]/3,3) )
#        boneKDTree = KDTree(dataPoints)
#        #find spine indices
#        t1 = time.time()
#        femurLeftindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
#        NN = boneKDTree.query_ball_tree(iso1KDTree, 1.0/scanData.sliceThickness)
#        for vindx in NN:
#            femurLeftindx[vindx] = True
#        print 'Took %f seconds to calculate %s vertices' % (time.time() - t1, bone)
#        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
#            savemat(kwargs['savePrefix']+'femurLeftvertices.mat', {'indx':femurLeftindx})

#    #Scale Left Femur
#    cummulativeJointTransform = atlasData.atlasJoint.getCummulativeTransform(id(jointOfInterest))
#    bones =['Upper Hindlimb Left']
# #    if 'Pelvis Hindlimb Left' not in fineAlignTransforms:    #Do ICP
#    if True:    #Do ICP
#        #step 1: get data vertex list. These vertexes are based on the current best transform
#        dataPoints = []
#        for bone in bones:
#            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
#            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
#        dataPoints = dataPoints.reshape( (dataPoints.shape[0]/3,3) )
#
#        #step 2: Filter model points as much as possible, ideally only bones to be matched will remain
#        #create filtered model point cloud
#        modelPoints = scanData.isosurfaceModel1.transformedVertexList.copy()
#        modelIndx = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)
#
#        #remove front half of model data
#        normV = scanData.alignmentAxis[0]
#        planePoint = (scanData.isosurfaceModel.maxPointTransformed + scanData.isosurfaceModel.minPointTransformed) / 2.0
#        pointVecs = modelPoints - planePoint
#        distance = numpy.dot(pointVecs, normV)
#        modelIndx[distance > 0] = False
#
#        #remove previously aligned vertecies
#        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(spineindx))
#        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tailindx))
#        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(skullindx))
#        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipLeftindx))
#        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipRightindx))
#
#        #Step 3: Reorient Point Clouds
#        #transform points back so that joint of interest is in original position/orientation
#        #the reason for this is that P=C*V then P = M*P, is not equal to P = (C*M)*V. Where C is parent transform, M is new ICP transform, V is original vector
#        #ICP must be done on original points not points midway through a transformation
#        modelPoints = numpyTransform.transformPoints(cummulativeJointTransform.I,modelPoints)
#        dataPoints = numpyTransform.transformPoints(cummulativeJointTransform.I,dataPoints)
#
#        #move point clouds so that joint is at origin
#        jointLocation = jointOfInterest.initialLocationMat[:3,3].getA().squeeze()
#        modelPoints -= jointLocation
#        dataPoints -= jointLocation
#
#        #Remove any points now that model is reoriented
#        modelIndx2 = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)
#
#        #Get just points of interest
#        modelPoints = modelPoints[numpy.logical_and(modelIndx, modelIndx2)]
#
#        #create different starting transformations
#        cummJointTransferRotOnly = numpy.matrix(numpy.identity(4))
#        cummJointTransferRotOnly[:3,:3] = cummulativeJointTransform[:3,:3]
#        proximodistalVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I,jointOfInterest.proximodistalVecTransformed)
#        secondaryVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I,jointOfInterest.secondaryVecTransformed)
#        initialTransforms = []
#        initialTransformsError = []
#        icpTransformsErr = []
#        icpTransforms = []
#        icpT = []
#
#        initialTransforms = [numpy.matrix(numpy.identity(4))]
#        #scaling transform
#        for initialTransform in initialTransforms:
#            dataPointsInitialTrandform = numpyTransform.transformPoints(initialTransform, dataPoints)
#
#            #Step 4: Perform actual ICP
#            transBound = jointOfInterest.DOFtrans / scanData.sliceThickness
#            print 'Scaling ICP'
#            icp = ICP(modelPoints, dataPointsInitialTrandform, maxIterations=100, modelDownsampleFactor=1, dataDownsampleFactor=1, minimizeMethod='fmincon')
#            initialGuess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
#            lowerBounds = numpy.array([-pi, -pi, -pi, -transBound, -transBound, -transBound, 0.8, 0.8, 0.8])
#            upperBounds = numpy.array([pi, pi, pi, transBound, transBound, transBound, 1.2, 1.2, 1.2])
#            transform, err, t = icp.runICP(x0=initialGuess, lb=lowerBounds,ub=upperBounds,scaleOnlyIso=True)
#            del icp
#            transform = transform[-1]
#            print 'ICP Generated Transform for %s Joint' % (jointOfInterest.name)
#            print transform
#
#            icpT.append(t)
#            icpTransformsErr.append(err)
#            icpTransforms.append(transform)
#
#            if visualize and False:    #display err plot of ICP
#                fig = plt.figure()
#                ax = fig.add_subplot(1,1,1)
#                ax.plot(t,err,'x--')
#                ax.set_xlabel('Time')
#                ax.set_ylabel('RMS error')
#                ax.set_title('Result of ICP on %s Joint, initial transform %d' %(jointOfInterest.name, len(initialTransformsError)))
#                plt.show()
#
#        icpTransformsErr = numpy.array(icpTransformsErr)
#
#        tindx = numpy.where(icpTransformsErr==icpTransformsErr.min())[0][0]
#        transform = icpTransforms[tindx] * initialTransforms[tindx]
#        print 'Best Scale iteration was %d with transform:' % (tindx)
#        print transform
#
#        if visualize:    #display err plot of ICP
#            fig = plt.figure()
#            ax = fig.add_subplot(1,1,1)
#            ax.plot(icpT[tindx],icpTransformsErr[tindx],'x--')
#            ax.set_xlabel('Time')
#            ax.set_ylabel('RMS error')
#            ax.set_title('Result of Scale ICP on %s Joint, initial transform %d' %(jointOfInterest.name, tindx))
#            plt.show()
#
#        if visualize and False:
#            #Display initial filtered points clouds that will be passed to ICP
#            tri = numpy.array(range(modelPoints.shape[0]-modelPoints.shape[0]%3))
#            tri = tri.reshape((tri.shape[0]/3,3))
#            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0,0.0,0.0])
#
#            tri = numpy.array(range(dataPoints.shape[0]-dataPoints.shape[0]%3))
#            tri = tri.reshape((tri.shape[0]/3,3))
#            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0,1.0,0.0])
#
#            dataPointsTransformed = numpyTransform.transformPoints(transform, dataPoints)
#            tri = numpy.array(range(dataPointsTransformed.shape[0]-dataPointsTransformed.shape[0]%3))
#            tri = tri.reshape((tri.shape[0]/3,3))
#            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0,0.0,1.0])
#
#            #display point clouds at final position
#            modelPoints += jointOfInterest.initialLocationMat[:3,3].getA().squeeze()
#            modelPoints = numpyTransform.transformPoints(cummulativeJointTransform, modelPoints)
#            tri = numpy.array(range(modelPoints.shape[0]-modelPoints.shape[0]%3))
#            tri = tri.reshape((tri.shape[0]/3,3))
#            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0,0.0,0.0])
#
#            dataPoints += jointOfInterest.initialLocationMat[:3,3].getA().squeeze()
#            dataPoints = numpyTransform.transformPoints(cummulativeJointTransform, dataPoints)
#            tri = numpy.array(range(dataPoints.shape[0]-dataPoints.shape[0]%3))
#            tri = tri.reshape((tri.shape[0]/3,3))
#            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0,1.0,0.0])
#
#            dataPointsTransformed += jointOfInterest.initialLocationMat[:3,3].getA().squeeze()
#            dataPointsTransformed = numpyTransform.transformPoints(cummulativeJointTransform, dataPointsTransformed)
#            tri = numpy.array(range(dataPointsTransformed.shape[0]-dataPointsTransformed.shape[0]%3))
#            tri = tri.reshape((tri.shape[0]/3,3))
#            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Final Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0,0.0,1.0])
#
#    else:    #skip ICP, just use predetermined transform
#        transform = numpy.matrix(fineAlignTransforms['Pelvis Hindlimb Left'])
#
#    #store transform
#    fineAlignTransforms[jointOfInterest.name] = transform
#
#    #Step 5: Apply ICP transform to joint
#    R = numpy.matrix(numpy.identity(4))
#    R[:3,:3] = transform[:3,:3]
#    jointOfInterest.rotate(R, relative=True)
#    cummulativeJointRotation = numpy.matrix(numpy.identity(4))
#    cummulativeJointRotation[:3,:3] = cummulativeJointTransform[:3,:3]
#    modifiedTranslate = cummulativeJointRotation*numpyTransform.translation(transform[:3,3].getA().squeeze())    #translation has to include rotation effects of cumulative transformations
# #    jointOfInterest.translate(modifiedTranslate[:3,3].getA().squeeze(), absolute=False)
#    jointOfInterest.translate(transform[:3,3].getA().squeeze(), absolute=False)

    if 'updateSceneFunc' in kwargs and kwargs['updateSceneFunc'] is not None:
        kwargs['updateSceneFunc']()
        time.sleep(3)

    # print resulting joint location
    if 'getResults' in kwargs and kwargs['getResults']:
        atlasData.atlasJoint.transformVertices()
        print '%s Joint Location: %s' % (jointOfInterest.name, str(jointOfInterest.location))
        meanDist = []
        for bone in bones:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            meanDist.append(numpy.sum(d) / len(i))
        print '%s Joint Mean surface to bone distance: %f' % (jointOfInterest.name, numpy.mean(meanDist))
        dist = 0.0
        numPoints = 0.0
        for bone in scanData.bonesVol1:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            dist += numpy.sum(d)
            numPoints += len(i)
        print 'Post %s Align Mean distance all bones to surface: %f' % (jointOfInterest.name, dist / numPoints)

    # get indices of model vertices within 1mm of aligned atlas hip left vertices
    if 'femurLeftVertexShortcut' in kwargs and os.path.exists(kwargs['femurLeftVertexShortcut']):
        vertexDict = loadmat(kwargs['femurLeftVertexShortcut'])
        femurLeftindx = vertexDict['indx'].astype(numpy.bool).squeeze()
    else:
        # create KDTree of newly aligned bone
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))
        boneKDTree = KDTree(dataPoints)
        # find spine indices
        t1 = time.time()
        femurLeftindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
        NN = boneKDTree.query_ball_tree(iso1KDTree, 1.0 / scanData.sliceThickness)
        for vindx in NN:
            femurLeftindx[vindx] = True
        print 'Took %f seconds to calculate %s vertices' % (time.time() - t1, bone)
        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
            savemat(kwargs['savePrefix'] + 'femurLeftvertices.mat', {'indx': femurLeftindx})

    ############################################################################################################################################
    # align Right Femur
    for joint in hipRight.childJoints:
        if joint.name == 'Pelvis Hindlimb Right':
            femurRight = joint
            break
    else:
        raise Exception('Correct Joint not found')
    jointOfInterest = femurRight
    cummulativeJointTransform = atlasData.atlasJoint.getCummulativeTransform(id(jointOfInterest))
    bones = ['Upper Hindlimb Right']
    if 'Pelvis Hindlimb Right' not in fineAlignTransforms:  # Do ICP
        # step 1: get data vertex list. These vertexes are based on the current best transform
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))

        # step 2: Filter model points as much as possible, ideally only bones to be matched will remain
        # create filtered model point cloud
        modelPoints = scanData.isosurfaceModel1.transformedVertexList.copy()
        modelIndx = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # remove front half of model data
        normV = scanData.alignmentAxis[0]
        planePoint = (scanData.isosurfaceModel.maxPointTransformed + scanData.isosurfaceModel.minPointTransformed) / 2.0
        pointVecs = modelPoints - planePoint
        distance = numpy.dot(pointVecs, normV)
        modelIndx[distance > 0] = False

        # remove previously aligned vertecies
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(spineindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tailindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(skullindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipRightindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(femurLeftindx))

        # Step 3: Reorient Point Clouds
        # transform points back so that joint of interest is in original position/orientation
        # the reason for this is that P=C*V then P = M*P, is not equal to P = (C*M)*V. Where C is parent transform, M is new ICP transform, V is original vector
        # ICP must be done on original points not points midway through a transformation
        modelPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, modelPoints)
        dataPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, dataPoints)

        # move point clouds so that joint is at origin
        jointLocation = jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
        modelPoints -= jointLocation
        dataPoints -= jointLocation

        # Remove any points now that model is reoriented
        modelIndx2 = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # Get just points of interest
        modelPoints = modelPoints[numpy.logical_and(modelIndx, modelIndx2)]

        # create different starting transformations
        cummJointTransferRotOnly = numpy.matrix(numpy.identity(4))
        cummJointTransferRotOnly[:3, :3] = cummulativeJointTransform[:3, :3]
        proximodistalVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.proximodistalVecTransformed)
        secondaryVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.secondaryVecTransformed)
        initialTransforms = []
        initialTransformsError = []
        icpTransformsErr = []
        icpTransforms = []
        icpT = []

        for spin in numpy.arange(0, 2 * numpy.pi, 3 * numpy.pi):
            for elevation in numpy.arange(0, numpy.pi, numpy.pi / 3):
                for aximuth in numpy.arange(0, 2 * numpy.pi, numpy.pi / 3):
                    initialTransforms.append(numpyTransform.rotation(aximuth, proximodistalVecTransformed, N=4) * numpyTransform.rotation(elevation, secondaryVecTransformed, N=4) * numpyTransform.rotation(spin, proximodistalVecTransformed, N=4))

        for i in xrange(len(initialTransforms)):
            initialTransform = initialTransforms[i]
            dataPointsInitialTrandform = numpyTransform.transformPoints(initialTransform, dataPoints)

            # limit translation
#            transBound = jointOfInterest.DOFtrans / (scanData.resampleFactor * scanData.sliceThickness)
            transBound = jointOfInterest.DOFtrans / scanData.sliceThickness
            print '%s translation limited to %fmm aka %f pixels' % (jointOfInterest.name, jointOfInterest.DOFtrans, transBound)
            # Step 4: Perform actual ICP
            print 'ITERATION', i
            icp = ICP(modelPoints, dataPointsInitialTrandform, maxIterations=15, modelDownsampleFactor=1, dataDownsampleFactor=1, minimizeMethod='fmincon')
            initialGuess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            lowerBounds = numpy.array([-pi, -pi, -pi, -transBound, -transBound, -transBound, 0.8, 0.8, 0.8])
            upperBounds = numpy.array([pi, pi, pi, transBound, transBound, transBound, 1.2, 1.2, 1.2])
            transform, err, t = icp.runICP(x0=initialGuess, lb=lowerBounds, ub=upperBounds)
            del icp
            transform = transform[-1]
            print 'ICP Generated Transform for %s Joint' % (jointOfInterest.name)
            print transform

            icpT.append(t)
            icpTransformsErr.append(err)
            icpTransforms.append(transform)

            if visualize and False:  # display err plot of ICP
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(t, err, 'x--')
                ax.set_xlabel('Time')
                ax.set_ylabel('RMS error')
                ax.set_title('Result of ICP on %s Joint, initial transform %d' % (jointOfInterest.name, len(initialTransformsError)))
                plt.show()

        icpTransformsErr = numpy.array(icpTransformsErr)

        tindx = numpy.where(icpTransformsErr == icpTransformsErr.min())[0][0]
        transform = icpTransforms[tindx] * initialTransforms[tindx]
        print 'Best iteration was %d with transform:' % (tindx)
        print transform

        if visualize:  # display err plot of ICP
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(icpT[tindx], icpTransformsErr[tindx], 'x--')
            ax.set_xlabel('Time')
            ax.set_ylabel('RMS error')
            ax.set_title('Result of ICP on %s Joint, initial transform %d' % (jointOfInterest.name, tindx))
            plt.show()

        if visualize and False:
            # Display initial filtered points clouds that will be passed to ICP
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed = numpyTransform.transformPoints(transform, dataPoints)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

            # display point clouds at final position
            modelPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            modelPoints = numpyTransform.transformPoints(cummulativeJointTransform, modelPoints)
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            dataPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPoints = numpyTransform.transformPoints(cummulativeJointTransform, dataPoints)
            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPointsTransformed = numpyTransform.transformPoints(cummulativeJointTransform, dataPointsTransformed)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Final Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

    else:  # skip ICP, just use predetermined transform
        transform = numpy.matrix(fineAlignTransforms['Pelvis Hindlimb Right'])
#        transform = numpy.matrix(
#            [[ 0.45810115,  0.3559261,   0.81453051,  8.06362508],
#             [ 0.88408411, -0.27769195, -0.37587559,  8.33369202],
#             [ 0.09240463,  0.89230252, -0.44187961, -6.87864907],
#             [ 0.,          0.,          0.,          1.,        ]])

    # store transform
    fineAlignTransforms[jointOfInterest.name] = transform

    # Step 5: Apply ICP transform to joint
    R = numpy.matrix(numpy.identity(4))
    R[:3, :3] = transform[:3, :3]
    jointOfInterest.rotate(R, relative=True)
    cummulativeJointRotation = numpy.matrix(numpy.identity(4))
    cummulativeJointRotation[:3, :3] = cummulativeJointTransform[:3, :3]
    modifiedTranslate = cummulativeJointRotation * numpyTransform.translation(transform[:3, 3].getA().squeeze())  # translation has to include rotation effects of cumulative transformations
#    jointOfInterest.translate(modifiedTranslate[:3,3].getA().squeeze(), absolute=False)
    jointOfInterest.translate(transform[:3, 3].getA().squeeze(), absolute=False)

    if 'updateSceneFunc' in kwargs and kwargs['updateSceneFunc'] is not None:
        kwargs['updateSceneFunc']()
        time.sleep(3)

    # print resulting joint location
    if 'getResults' in kwargs and kwargs['getResults']:
        atlasData.atlasJoint.transformVertices()
        print '%s Joint Location: %s' % (jointOfInterest.name, str(jointOfInterest.location))
        meanDist = []
        for bone in bones:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            meanDist.append(numpy.sum(d) / len(i))
        print '%s Joint Mean surface to bone distance: %f' % (jointOfInterest.name, numpy.mean(meanDist))
        dist = 0.0
        numPoints = 0.0
        for bone in scanData.bonesVol1:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            dist += numpy.sum(d)
            numPoints += len(i)
        print 'Post %s Align Mean distance all bones to surface: %f' % (jointOfInterest.name, dist / numPoints)

    # get indices of model vertices within 1mm of aligned atlas hip left vertices
    if 'femurRightVertexShortcut' in kwargs and os.path.exists(kwargs['femurRightVertexShortcut']):
        vertexDict = loadmat(kwargs['femurRightVertexShortcut'])
        femurRightindx = vertexDict['indx'].astype(numpy.bool).squeeze()
    else:
        # create KDTree of newly aligned bone
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))
        boneKDTree = KDTree(dataPoints)
        # find spine indices
        t1 = time.time()
        femurRightindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
        NN = boneKDTree.query_ball_tree(iso1KDTree, 1.0 / scanData.sliceThickness)
        for vindx in NN:
            femurRightindx[vindx] = True
        print 'Took %f seconds to calculate %s vertices' % (time.time() - t1, bone)
        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
            savemat(kwargs['savePrefix'] + 'femurRightvertices.mat', {'indx': femurRightindx})

    ############################################################################################################################################
    # align Left Tibia
    for joint in femurLeft.childJoints:
        if joint.name == 'Hindlimb Left Knee':
            tibiaLeft = joint
            break
    else:
        raise Exception('Correct Joint not found')
    jointOfInterest = tibiaLeft
    cummulativeJointTransform = atlasData.atlasJoint.getCummulativeTransform(id(jointOfInterest))
    bones = ['Lower Hindlimb Left']
    if 'Hindlimb Left Knee' not in fineAlignTransforms:  # Do ICP
#    if True:
        # step 1: get data vertex list. These vertexes are based on the current best transform
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))

        # step 2: Filter model points as much as possible, ideally only bones to be matched will remain
        # create filtered model point cloud
        modelPoints = scanData.isosurfaceModel1.transformedVertexList.copy()
        modelIndx = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # remove front half of model data
        normV = scanData.alignmentAxis[0]
        planePoint = (scanData.isosurfaceModel.maxPointTransformed + scanData.isosurfaceModel.minPointTransformed) / 2.0
        pointVecs = modelPoints - planePoint
        distance = numpy.dot(pointVecs, normV)
        modelIndx[distance > 0] = False

        # remove previously aligned vertecies
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(spineindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tailindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(skullindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipRightindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(femurLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(femurRightindx))

        # Step 3: Reorient Point Clouds
        # transform points back so that joint of interest is in original position/orientation
        # the reason for this is that P=C*V then P = M*P, is not equal to P = (C*M)*V. Where C is parent transform, M is new ICP transform, V is original vector
        # ICP must be done on original points not points midway through a transformation
        modelPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, modelPoints)
        dataPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, dataPoints)

        # move point clouds so that joint is at origin
        jointLocation = jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
        modelPoints -= jointLocation
        dataPoints -= jointLocation

        # Remove any points now that model is reoriented
        modelIndx2 = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # Get just points of interest
        modelPoints = modelPoints[numpy.logical_and(modelIndx, modelIndx2)]

        # create different starting transformations
        cummJointTransferRotOnly = numpy.matrix(numpy.identity(4))
        cummJointTransferRotOnly[:3, :3] = cummulativeJointTransform[:3, :3]
        proximodistalVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.proximodistalVecTransformed)
        secondaryVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.secondaryVecTransformed)
        initialTransforms = []
        initialTransformsError = []
        icpTransformsErr = []
        icpTransforms = []
        icpT = []

        for spin in numpy.arange(0, 2 * numpy.pi, 3 * numpy.pi):
            for elevation in numpy.arange(0, numpy.pi, numpy.pi / 4):
                for aximuth in numpy.arange(0, 2 * numpy.pi, numpy.pi / 4):
                    initialTransforms.append(numpyTransform.rotation(aximuth, proximodistalVecTransformed, N=4) * numpyTransform.rotation(elevation, secondaryVecTransformed, N=4) * numpyTransform.rotation(spin, proximodistalVecTransformed, N=4))

#        initialTransforms = initialTransforms[24:26]

        for i in xrange(len(initialTransforms)):
            initialTransform = initialTransforms[i]
            dataPointsInitialTrandform = numpyTransform.transformPoints(initialTransform, dataPoints)

            # limit translation
#            transBound = jointOfInterest.DOFtrans / (scanData.resampleFactor * scanData.sliceThickness)
            transBound = jointOfInterest.DOFtrans / scanData.sliceThickness
            print '%s translation limited to %fmm aka %f pixels' % (jointOfInterest.name, jointOfInterest.DOFtrans, transBound)
            # Step 4: Perform actual ICP
            print 'ITERATION', i
            icp = ICP(modelPoints, dataPointsInitialTrandform, maxIterations=15, modelDownsampleFactor=1, dataDownsampleFactor=1, minimizeMethod='fmincon')
            initialGuess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            lowerBounds = numpy.array([-pi, -pi, -pi, -transBound, -transBound, -transBound, 0.8, 0.8, 0.8])
            upperBounds = numpy.array([pi, pi, pi, transBound, transBound, transBound, 1.2, 1.2, 1.2])
            transform, err, t = icp.runICP(x0=initialGuess, lb=lowerBounds, ub=upperBounds)
            del icp
            transform = transform[-1]
            print 'ICP Generated Transform for %s Joint' % (jointOfInterest.name)
            print transform

            if transform[0, 3] > upperBounds[3] or transform[0, 3] < lowerBounds[3]:
                print 'x translation incorrect'
            if transform[1, 3] > upperBounds[4] or transform[1, 3] < lowerBounds[4]:
                print 'Y translation incorrect'
            if transform[2, 3] > upperBounds[5] or transform[2, 3] < lowerBounds[5]:
                print 'Z translation incorrect'

            icpT.append(t)
            icpTransformsErr.append(err)
            icpTransforms.append(transform)

            if visualize and False:  # display err plot of ICP
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(t, err, 'x--')
                ax.set_xlabel('Time')
                ax.set_ylabel('RMS error')
                ax.set_title('Result of ICP on %s Joint, initial transform %d' % (jointOfInterest.name, i))
                plt.show()

        icpTransformsErr = numpy.array(icpTransformsErr)

#        tindx = numpy.where(icpTransformsErr==icpTransformsErr.min())[0][0]
        tindx = numpy.where(icpTransformsErr == icpTransformsErr[:, -1].min())[0][0]
        transform = icpTransforms[tindx] * initialTransforms[tindx]
        print 'Best iteration was %d with transform:' % (tindx)
        print transform

        if visualize:  # display err plot of ICP
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(icpT[tindx], icpTransformsErr[tindx], 'x--')
            ax.set_xlabel('Time')
            ax.set_ylabel('RMS error')
            ax.set_title('Result of ICP on %s Joint, initial transform %d' % (jointOfInterest.name, tindx))
            plt.show()

        if visualize and False:
            # Display initial filtered points clouds that will be passed to ICP
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed = numpyTransform.transformPoints(transform, dataPoints)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

            # display point clouds at final position
            modelPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            modelPoints = numpyTransform.transformPoints(cummulativeJointTransform, modelPoints)
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            dataPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPoints = numpyTransform.transformPoints(cummulativeJointTransform, dataPoints)
            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPointsTransformed = numpyTransform.transformPoints(cummulativeJointTransform, dataPointsTransformed)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Final Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

    else:  # skip ICP, just use predetermined transform
        transform = numpy.matrix(fineAlignTransforms['Hindlimb Left Knee'])
#        transform = numpy.matrix(
#                [[  5.65588121e-01,   7.25237403e-01,   3.92607675e-01,  -2.81949076e+01],
#                 [ -2.21102136e-01,  -3.25288973e-01,   9.19402485e-01,   2.16992939e+01],
#                 [  7.94496018e-01,  -6.06809520e-01,  -2.36280401e-02,  -2.50706200e+01],
#                 [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    # store transform
    fineAlignTransforms[jointOfInterest.name] = transform

    # Step 5: Apply ICP transform to joint
    R = numpy.matrix(numpy.identity(4))
    R[:3, :3] = transform[:3, :3]
    jointOfInterest.rotate(R, relative=True)
    cummulativeJointRotation = numpy.matrix(numpy.identity(4))
    cummulativeJointRotation[:3, :3] = cummulativeJointTransform[:3, :3]
    modifiedTranslate = cummulativeJointRotation * numpyTransform.translation(transform[:3, 3].getA().squeeze())  # translation has to include rotation effects of cumulative transformations
#    jointOfInterest.translate(modifiedTranslate[:3,3].getA().squeeze(), absolute=False)
    jointOfInterest.translate(transform[:3, 3].getA().squeeze(), absolute=False)

    if 'updateSceneFunc' in kwargs and kwargs['updateSceneFunc'] is not None:
        kwargs['updateSceneFunc']()
        time.sleep(3)

    # print resulting joint location
    if 'getResults' in kwargs and kwargs['getResults']:
        atlasData.atlasJoint.transformVertices()
        print '%s Joint Location: %s' % (jointOfInterest.name, str(jointOfInterest.location))
        meanDist = []
        for bone in bones:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            meanDist.append(numpy.sum(d) / len(i))
        print '%s Joint Mean surface to bone distance: %f' % (jointOfInterest.name, numpy.mean(meanDist))
        dist = 0.0
        numPoints = 0.0
        for bone in scanData.bonesVol1:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            dist += numpy.sum(d)
            numPoints += len(i)
        print 'Post %s Align Mean distance all bones to surface: %f' % (jointOfInterest.name, dist / numPoints)

#    #rotote femur and tibia so that tibia hinge joint lines up with secondary axis
#    atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels['Upper Hindlimb Left']))
#    atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels['Lower Hindlimb Left']))
#    sv = numpy.cross(femurLeft.proximodistalVecTransformed, tibiaLeft.proximodistalVecTransformed)
#    tv = numpy.cross(tibiaLeft.proximodistalVecTransformed, sv)
#    coordAlignTform = numpyTransform.coordinateSystemConversionMatrix([tibiaLeft.proximodistalVecTransformed, tibiaLeft.secondaryVecTransformed, tibiaLeft.tertiaryVecTransformed], [tibiaLeft.proximodistalVecTransformed, sv, tv], N=4)
#    jointOfInterest.rotate(coordAlignTform, relative=True)

    # get indices of model vertices within 1mm of aligned atlas hip left vertices
    if 'tibiaLeftVertexShortcut' in kwargs and os.path.exists(kwargs['tibiaLeftVertexShortcut']):
        vertexDict = loadmat(kwargs['tibiaLeftVertexShortcut'])
        tibiaLeftindx = vertexDict['indx'].astype(numpy.bool).squeeze()
    else:
        # create KDTree of newly aligned bone
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))
        boneKDTree = KDTree(dataPoints)
        # find spine indices
        t1 = time.time()
        tibiaLeftindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
        NN = boneKDTree.query_ball_tree(iso1KDTree, 1.0 / scanData.sliceThickness)
        for vindx in NN:
            tibiaLeftindx[vindx] = True
        print 'Took %f seconds to calculate %s vertices' % (time.time() - t1, bone)
        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
            savemat(kwargs['savePrefix'] + 'tibiaLeftvertices.mat', {'indx': tibiaLeftindx})

    ############################################################################################################################################
    # align Right Tibia
    for joint in femurRight.childJoints:
        if joint.name == 'Hindlimb Right Knee':
            tibiaRight = joint
            break
    else:
        raise Exception('Correct Joint not found')
    jointOfInterest = tibiaRight
    cummulativeJointTransform = atlasData.atlasJoint.getCummulativeTransform(id(jointOfInterest))
    bones = ['Lower Hindlimb Right']
    if 'Hindlimb Right Knee' not in fineAlignTransforms:  # Do ICP
        # step 1: get data vertex list. These vertexes are based on the current best transform
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))

        # step 2: Filter model points as much as possible, ideally only bones to be matched will remain
        # create filtered model point cloud
        modelPoints = scanData.isosurfaceModel1.transformedVertexList.copy()
        modelIndx = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # remove front half of model data
        normV = scanData.alignmentAxis[0]
        planePoint = (scanData.isosurfaceModel.maxPointTransformed + scanData.isosurfaceModel.minPointTransformed) / 2.0
        pointVecs = modelPoints - planePoint
        distance = numpy.dot(pointVecs, normV)
        modelIndx[distance > 0] = False

        # remove previously aligned vertecies
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(spineindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tailindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(skullindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipRightindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(femurLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(femurRightindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tibiaLeftindx))

        # Step 3: Reorient Point Clouds
        # transform points back so that joint of interest is in original position/orientation
        # the reason for this is that P=C*V then P = M*P, is not equal to P = (C*M)*V. Where C is parent transform, M is new ICP transform, V is original vector
        # ICP must be done on original points not points midway through a transformation
        modelPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, modelPoints)
        dataPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, dataPoints)

        # move point clouds so that joint is at origin
        jointLocation = jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
        modelPoints -= jointLocation
        dataPoints -= jointLocation

        # Remove any points now that model is reoriented
        modelIndx2 = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # Get just points of interest
        modelPoints = modelPoints[numpy.logical_and(modelIndx, modelIndx2)]

        # create different starting transformations
        cummJointTransferRotOnly = numpy.matrix(numpy.identity(4))
        cummJointTransferRotOnly[:3, :3] = cummulativeJointTransform[:3, :3]
        proximodistalVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.proximodistalVecTransformed)
        secondaryVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.secondaryVecTransformed)
        initialTransforms = []
        initialTransformsError = []
        icpTransformsErr = []
        icpTransforms = []
        icpT = []

        for spin in numpy.arange(0, 2 * numpy.pi, 3 * numpy.pi):
            for elevation in numpy.arange(0, numpy.pi, numpy.pi / 4):
                for aximuth in numpy.arange(0, 2 * numpy.pi, numpy.pi / 4):
                    initialTransforms.append(numpyTransform.rotation(aximuth, proximodistalVecTransformed, N=4) * numpyTransform.rotation(elevation, secondaryVecTransformed, N=4) * numpyTransform.rotation(spin, proximodistalVecTransformed, N=4))

        for i in xrange(len(initialTransforms)):
            initialTransform = initialTransforms[i]
            dataPointsInitialTrandform = numpyTransform.transformPoints(initialTransform, dataPoints)

            # limit translation
#            transBound = jointOfInterest.DOFtrans / (scanData.resampleFactor * scanData.sliceThickness)
            transBound = jointOfInterest.DOFtrans / scanData.sliceThickness
            print '%s translation limited to %fmm aka %f pixels' % (jointOfInterest.name, jointOfInterest.DOFtrans, transBound)
            # Step 4: Perform actual ICP
            print 'ITERATION', i
            icp = ICP(modelPoints, dataPointsInitialTrandform, maxIterations=15, modelDownsampleFactor=1, dataDownsampleFactor=1, minimizeMethod='fmincon')
            initialGuess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            lowerBounds = numpy.array([-pi, -pi, -pi, -transBound, -transBound, -transBound, 0.8, 0.8, 0.8])
            upperBounds = numpy.array([pi, pi, pi, transBound, transBound, transBound, 1.2, 1.2, 1.2])
            transform, err, t = icp.runICP(x0=initialGuess, lb=lowerBounds, ub=upperBounds)
            del icp
            transform = transform[-1]
            print 'ICP Generated Transform for %s Joint' % (jointOfInterest.name)
            print transform

            if transform[0, 3] > upperBounds[3] or transform[0, 3] < lowerBounds[3]:
                print 'x translation incorrect'
            if transform[1, 3] > upperBounds[4] or transform[1, 3] < lowerBounds[4]:
                print 'Y translation incorrect'
            if transform[2, 3] > upperBounds[5] or transform[2, 3] < lowerBounds[5]:
                print 'Z translation incorrect'

            icpT.append(t)
            icpTransformsErr.append(err)
            icpTransforms.append(transform)

            if visualize and False:  # display err plot of ICP
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(t, err, 'x--')
                ax.set_xlabel('Time')
                ax.set_ylabel('RMS error')
                ax.set_title('Result of ICP on %s Joint, initial transform %d' % (jointOfInterest.name, i))
                plt.show()

        icpTransformsErr = numpy.array(icpTransformsErr)

#        tindx = numpy.where(icpTransformsErr==icpTransformsErr.min())[0][0]
        tindx = numpy.where(icpTransformsErr == icpTransformsErr[:, -1].min())[0][0]
        transform = icpTransforms[tindx] * initialTransforms[tindx]
        print 'Best iteration was %d with transform:' % (tindx)
        print transform

        if visualize:  # display err plot of ICP
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(icpT[tindx], icpTransformsErr[tindx], 'x--')
            ax.set_xlabel('Time')
            ax.set_ylabel('RMS error')
            ax.set_title('Result of ICP on %s Joint, initial transform %d' % (jointOfInterest.name, tindx))
            plt.show()

        if visualize and False:
            # Display initial filtered points clouds that will be passed to ICP
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed = numpyTransform.transformPoints(transform, dataPoints)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

            # display point clouds at final position
            modelPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            modelPoints = numpyTransform.transformPoints(cummulativeJointTransform, modelPoints)
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            dataPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPoints = numpyTransform.transformPoints(cummulativeJointTransform, dataPoints)
            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPointsTransformed = numpyTransform.transformPoints(cummulativeJointTransform, dataPointsTransformed)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Final Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

    else:  # skip ICP, just use predetermined transform
        transform = numpy.matrix(fineAlignTransforms['Hindlimb Right Knee'])
#        transform = numpy.matrix(
#            [[  0.61328114,  -0.74659998,  -0.2578269,   25.08829485],
#             [  0.23768732,  -0.13684834,   0.9616534,    9.49064482],
#             [ -0.7532536,   -0.65104608,   0.09353084, -26.06480535],
#             [  0.,           0.,           0.,           1.        ]])

    # store transform
    fineAlignTransforms[jointOfInterest.name] = transform

    # Step 5: Apply ICP transform to joint
    R = numpy.matrix(numpy.identity(4))
    R[:3, :3] = transform[:3, :3]
    jointOfInterest.rotate(R, relative=True)
    cummulativeJointRotation = numpy.matrix(numpy.identity(4))
    cummulativeJointRotation[:3, :3] = cummulativeJointTransform[:3, :3]
    modifiedTranslate = cummulativeJointRotation * numpyTransform.translation(transform[:3, 3].getA().squeeze())  # translation has to include rotation effects of cumulative transformations
#    jointOfInterest.translate(modifiedTranslate[:3,3].getA().squeeze(), absolute=False)
    jointOfInterest.translate(transform[:3, 3].getA().squeeze(), absolute=False)

    if 'updateSceneFunc' in kwargs and kwargs['updateSceneFunc'] is not None:
        kwargs['updateSceneFunc']()
        time.sleep(3)

    # print resulting joint location
    if 'getResults' in kwargs and kwargs['getResults']:
        atlasData.atlasJoint.transformVertices()
        print '%s Joint Location: %s' % (jointOfInterest.name, str(jointOfInterest.location))
        meanDist = []
        for bone in bones:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            meanDist.append(numpy.sum(d) / len(i))
        print '%s Joint Mean surface to bone distance: %f' % (jointOfInterest.name, numpy.mean(meanDist))
        dist = 0.0
        numPoints = 0.0
        for bone in scanData.bonesVol1:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            dist += numpy.sum(d)
            numPoints += len(i)
        print 'Post %s Align Mean distance all bones to surface: %f' % (jointOfInterest.name, dist / numPoints)

#    #rotote femur and tibia so that tibia hinge joint lines up with secondary axis
#    atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels['Upper Hindlimb Right']))
#    atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels['Lower Hindlimb Right']))
#    sv = numpy.cross(femurRight.proximodistalVecTransformed, tibiaRight.proximodistalVecTransformed)
#    tv = numpy.cross(tibiaRight.proximodistalVecTransformed, sv)
#    coordAlignTform = numpyTransform.coordinateSystemConversionMatrix([tibiaRight.proximodistalVecTransformed, tibiaRight.secondaryVecTransformed, tibiaRight.tertiaryVecTransformed], [tibiaRight.proximodistalVecTransformed, sv, tv], N=4)
#    jointOfInterest.rotate(coordAlignTform, relative=True)

    # get indices of model vertices within 1mm of aligned atlas hip left vertices
    if 'tibiaRightVertexShortcut' in kwargs and os.path.exists(kwargs['tibiaRightVertexShortcut']):
        vertexDict = loadmat(kwargs['tibiaRightVertexShortcut'])
        tibiaRightindx = vertexDict['indx'].astype(numpy.bool).squeeze()
    else:
        # create KDTree of newly aligned bone
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))
        boneKDTree = KDTree(dataPoints)
        # find spine indices
        t1 = time.time()
        tibiaRightindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
        NN = boneKDTree.query_ball_tree(iso1KDTree, 1.0 / scanData.sliceThickness)
        for vindx in NN:
            tibiaRightindx[vindx] = True
        print 'Took %f seconds to calculate %s vertices' % (time.time() - t1, bone)
        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
            savemat(kwargs['savePrefix'] + 'tibiaRightvertices.mat', {'indx': tibiaRightindx})

    ############################################################################################################################################
    # align Lower Left Paw
    for joint in tibiaLeft.childJoints:
        if joint.name == 'Hindlimb Left Ankle':
            ankleLeft = joint
            break
    else:
        raise Exception('Correct Joint not found')
    jointOfInterest = ankleLeft
    cummulativeJointTransform = atlasData.atlasJoint.getCummulativeTransform(id(jointOfInterest))
    bones = ['HindPaw Left']
    if 'Hindlimb Left Ankle' not in fineAlignTransforms:  # Do ICP
        # step 1: get data vertex list. These vertexes are based on the current best transform
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))

        # step 2: Filter model points as much as possible, ideally only bones to be matched will remain
        # create filtered model point cloud
        modelPoints = scanData.isosurfaceModel1.transformedVertexList.copy()
        modelIndx = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # remove front half of model data
        normV = scanData.alignmentAxis[0]
        planePoint = (scanData.isosurfaceModel.maxPointTransformed + scanData.isosurfaceModel.minPointTransformed) / 2.0
        pointVecs = modelPoints - planePoint
        distance = numpy.dot(pointVecs, normV)
        modelIndx[distance > 0] = False

        # remove previously aligned vertecies
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(spineindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tailindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(skullindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipRightindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(femurLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(femurRightindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tibiaLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tibiaRightindx))

        # Step 3: Reorient Point Clouds
        # transform points back so that joint of interest is in original position/orientation
        # the reason for this is that P=C*V then P = M*P, is not equal to P = (C*M)*V. Where C is parent transform, M is new ICP transform, V is original vector
        # ICP must be done on original points not points midway through a transformation
        modelPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, modelPoints)
        dataPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, dataPoints)

        # move point clouds so that joint is at origin
        jointLocation = jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
        modelPoints -= jointLocation
        dataPoints -= jointLocation

        # Remove any points now that model is reoriented
        modelIndx2 = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # Get just points of interest
        modelPoints = modelPoints[numpy.logical_and(modelIndx, modelIndx2)]

        # create different starting transformations
        cummJointTransferRotOnly = numpy.matrix(numpy.identity(4))
        cummJointTransferRotOnly[:3, :3] = cummulativeJointTransform[:3, :3]
        proximodistalVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.proximodistalVecTransformed)
        secondaryVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.secondaryVecTransformed)
        initialTransforms = []
        initialTransformsError = []
        icpTransformsErr = []
        icpTransforms = []
        icpT = []

        for spin in numpy.arange(0, 2 * numpy.pi, 3 * numpy.pi):
            for elevation in numpy.arange(0, numpy.pi, numpy.pi / 2):
                for aximuth in numpy.arange(0, 2 * numpy.pi, numpy.pi):
                    initialTransforms.append(numpyTransform.rotation(aximuth, proximodistalVecTransformed, N=4) * numpyTransform.rotation(elevation, secondaryVecTransformed, N=4) * numpyTransform.rotation(spin, proximodistalVecTransformed, N=4))

        for i in xrange(len(initialTransforms)):
            initialTransform = initialTransforms[i]
            dataPointsInitialTrandform = numpyTransform.transformPoints(initialTransform, dataPoints)

            # limit translation
#            transBound = jointOfInterest.DOFtrans / (scanData.resampleFactor * scanData.sliceThickness)
            transBound = jointOfInterest.DOFtrans / scanData.sliceThickness
            print '%s translation limited to %fmm aka %f pixels' % (jointOfInterest.name, jointOfInterest.DOFtrans, transBound)
            # Step 4: Perform actual ICP
            print 'ITERATION', i
            icp = ICP(modelPoints, dataPointsInitialTrandform, maxIterations=15, modelDownsampleFactor=1, dataDownsampleFactor=1, minimizeMethod='fmincon')
            initialGuess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            lowerBounds = numpy.array([-pi, -pi, -pi, -transBound, -transBound, -transBound, 0.8, 0.8, 0.8])
            upperBounds = numpy.array([pi, pi, pi, transBound, transBound, transBound, 1.2, 1.2, 1.2])
            transform, err, t = icp.runICP(x0=initialGuess, lb=lowerBounds, ub=upperBounds)
            del icp
            transform = transform[-1]
            print 'ICP Generated Transform for %s Joint' % (jointOfInterest.name)
            print transform

            if transform[0, 3] > upperBounds[3] or transform[0, 3] < lowerBounds[3]:
                print 'x translation incorrect'
            if transform[1, 3] > upperBounds[4] or transform[1, 3] < lowerBounds[4]:
                print 'Y translation incorrect'
            if transform[2, 3] > upperBounds[5] or transform[2, 3] < lowerBounds[5]:
                print 'Z translation incorrect'

            icpT.append(t)
            icpTransformsErr.append(err)
            icpTransforms.append(transform)

            if visualize and False:  # display err plot of ICP
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(t, err, 'x--')
                ax.set_xlabel('Time')
                ax.set_ylabel('RMS error')
                ax.set_title('Result of ICP on %s Joint, initial transform %d' % (jointOfInterest.name, i))
                plt.show()

        icpTransformsErr = numpy.array(icpTransformsErr)

        tindx = numpy.where(icpTransformsErr == icpTransformsErr[:, -1].min())[0][0]
        transform = icpTransforms[tindx] * initialTransforms[tindx]
        print 'Best iteration was %d with transform:' % (tindx)
        print transform

        if visualize:  # display err plot of ICP
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(icpT[tindx], icpTransformsErr[tindx], 'x--')
            ax.set_xlabel('Time')
            ax.set_ylabel('RMS error')
            ax.set_title('Result of ICP on %s Joint, initial transform %d' % (jointOfInterest.name, tindx))
            plt.show()

        if visualize and False:
            # Display initial filtered points clouds that will be passed to ICP
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed = numpyTransform.transformPoints(transform, dataPoints)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

            # display point clouds at final position
            modelPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            modelPoints = numpyTransform.transformPoints(cummulativeJointTransform, modelPoints)
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            dataPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPoints = numpyTransform.transformPoints(cummulativeJointTransform, dataPoints)
            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPointsTransformed = numpyTransform.transformPoints(cummulativeJointTransform, dataPointsTransformed)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Final Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

    else:  # skip ICP, just use predetermined transform
        transform = numpy.matrix(fineAlignTransforms['Hindlimb Left Ankle'])
#        transform = numpy.matrix(
#            [[  0.8903688,    0.0683342,   -0.45008203,   7.15709156],
#             [  0.11739263,   0.92076859,   0.37202711,  18.74429597],
#             [  0.43984357,  -0.38407764,   0.8118017,   10.02141994],
#             [  0.,           0.,           0.,           1.        ]])

    # store transform
    fineAlignTransforms[jointOfInterest.name] = transform

    # Step 5: Apply ICP transform to joint
    R = numpy.matrix(numpy.identity(4))
    R[:3, :3] = transform[:3, :3]
    jointOfInterest.rotate(R, relative=True)
    cummulativeJointRotation = numpy.matrix(numpy.identity(4))
    cummulativeJointRotation[:3, :3] = cummulativeJointTransform[:3, :3]
    modifiedTranslate = cummulativeJointRotation * numpyTransform.translation(transform[:3, 3].getA().squeeze())  # translation has to include rotation effects of cumulative transformations
#    jointOfInterest.translate(modifiedTranslate[:3,3].getA().squeeze(), absolute=False)
    jointOfInterest.translate(transform[:3, 3].getA().squeeze(), absolute=False)

    if 'updateSceneFunc' in kwargs and kwargs['updateSceneFunc'] is not None:
        kwargs['updateSceneFunc']()
        time.sleep(3)

    # print resulting joint location
    if 'getResults' in kwargs and kwargs['getResults']:
        atlasData.atlasJoint.transformVertices()
        print '%s Joint Location: %s' % (jointOfInterest.name, str(jointOfInterest.location))
        meanDist = []
        for bone in bones:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            meanDist.append(numpy.sum(d) / len(i))
        print '%s Joint Mean surface to bone distance: %f' % (jointOfInterest.name, numpy.mean(meanDist))
        dist = 0.0
        numPoints = 0.0
        for bone in scanData.bonesVol1:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            dist += numpy.sum(d)
            numPoints += len(i)
        print 'Post %s Align Mean distance all bones to surface: %f' % (jointOfInterest.name, dist / numPoints)

    # get indices of model vertices within 1mm of aligned atlas hip left vertices
    if 'lowerLeftPawVertexShortcut' in kwargs and os.path.exists(kwargs['lowerLeftPawVertexShortcut']):
        vertexDict = loadmat(kwargs['lowerLeftPawVertexShortcut'])
        lowerLeftPawindx = vertexDict['indx'].astype(numpy.bool).squeeze()
    else:
        # create KDTree of newly aligned bone
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))
        boneKDTree = KDTree(dataPoints)
        # find spine indices
        t1 = time.time()
        lowerLeftPawindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
        NN = boneKDTree.query_ball_tree(iso1KDTree, 1.0 / scanData.sliceThickness)
        for vindx in NN:
            lowerLeftPawindx[vindx] = True
        print 'Took %f seconds to calculate %s vertices' % (time.time() - t1, bone)
        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
            savemat(kwargs['savePrefix'] + 'lowerLeftPawvertices.mat', {'indx': lowerLeftPawindx})

    ############################################################################################################################################
    # align Lower Right Paw
    for joint in tibiaRight.childJoints:
        if joint.name == 'Hindlimb Right Ankle':
            ankleRight = joint
            break
    else:
        raise Exception('Correct Joint not found')
    jointOfInterest = ankleRight
    cummulativeJointTransform = atlasData.atlasJoint.getCummulativeTransform(id(jointOfInterest))
    bones = ['HindPaw Right']
    if 'Hindlimb Right Ankle' not in fineAlignTransforms:  # Do ICP
        # step 1: get data vertex list. These vertexes are based on the current best transform
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))

        # step 2: Filter model points as much as possible, ideally only bones to be matched will remain
        # create filtered model point cloud
        modelPoints = scanData.isosurfaceModel1.transformedVertexList.copy()
        modelIndx = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # remove front half of model data
        normV = scanData.alignmentAxis[0]
        planePoint = (scanData.isosurfaceModel.maxPointTransformed + scanData.isosurfaceModel.minPointTransformed) / 2.0
        pointVecs = modelPoints - planePoint
        distance = numpy.dot(pointVecs, normV)
        modelIndx[distance > 0] = False

        # remove previously aligned vertecies
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(spineindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tailindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(skullindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(hipRightindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(femurLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(femurRightindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tibiaLeftindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(tibiaRightindx))
        modelIndx = numpy.logical_and(modelIndx, numpy.logical_not(lowerLeftPawindx))

        # Step 3: Reorient Point Clouds
        # transform points back so that joint of interest is in original position/orientation
        # the reason for this is that P=C*V then P = M*P, is not equal to P = (C*M)*V. Where C is parent transform, M is new ICP transform, V is original vector
        # ICP must be done on original points not points midway through a transformation
        modelPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, modelPoints)
        dataPoints = numpyTransform.transformPoints(cummulativeJointTransform.I, dataPoints)

        # move point clouds so that joint is at origin
        jointLocation = jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
        modelPoints -= jointLocation
        dataPoints -= jointLocation

        # Remove any points now that model is reoriented
        modelIndx2 = numpy.ones(modelPoints.shape[0], dtype=numpy.bool)

        # Get just points of interest
        modelPoints = modelPoints[numpy.logical_and(modelIndx, modelIndx2)]

        # create different starting transformations
        cummJointTransferRotOnly = numpy.matrix(numpy.identity(4))
        cummJointTransferRotOnly[:3, :3] = cummulativeJointTransform[:3, :3]
        proximodistalVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.proximodistalVecTransformed)
        secondaryVecTransformed = numpyTransform.transformPoints(cummJointTransferRotOnly.I, jointOfInterest.secondaryVecTransformed)
        initialTransforms = []
        initialTransformsError = []
        icpTransformsErr = []
        icpTransforms = []
        icpT = []

        for spin in numpy.arange(0, 2 * numpy.pi, 3 * numpy.pi):
            for elevation in numpy.arange(0, numpy.pi, numpy.pi / 2):
                for aximuth in numpy.arange(0, 2 * numpy.pi, numpy.pi):
                    initialTransforms.append(numpyTransform.rotation(aximuth, proximodistalVecTransformed, N=4) * numpyTransform.rotation(elevation, secondaryVecTransformed, N=4) * numpyTransform.rotation(spin, proximodistalVecTransformed, N=4))

        for i in xrange(len(initialTransforms)):
            initialTransform = initialTransforms[i]
            dataPointsInitialTrandform = numpyTransform.transformPoints(initialTransform, dataPoints)

            # limit translation
#            transBound = jointOfInterest.DOFtrans / (scanData.resampleFactor * scanData.sliceThickness)
            transBound = jointOfInterest.DOFtrans / scanData.sliceThickness
            print '%s translation limited to %fmm aka %f pixels' % (jointOfInterest.name, jointOfInterest.DOFtrans, transBound)
            # Step 4: Perform actual ICP
            print 'ITERATION', i
            icp = ICP(modelPoints, dataPointsInitialTrandform, maxIterations=15, modelDownsampleFactor=1, dataDownsampleFactor=1, minimizeMethod='fmincon')
            initialGuess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            lowerBounds = numpy.array([-pi, -pi, -pi, -transBound, -transBound, -transBound, 0.8, 0.8, 0.8])
            upperBounds = numpy.array([pi, pi, pi, transBound, transBound, transBound, 1.2, 1.2, 1.2])
            transform, err, t = icp.runICP(x0=initialGuess, lb=lowerBounds, ub=upperBounds)
            del icp
            transform = transform[-1]
            print 'ICP Generated Transform for %s Joint' % (jointOfInterest.name)
            print transform

            if transform[0, 3] > upperBounds[3] or transform[0, 3] < lowerBounds[3]:
                print 'x translation incorrect'
            if transform[1, 3] > upperBounds[4] or transform[1, 3] < lowerBounds[4]:
                print 'Y translation incorrect'
            if transform[2, 3] > upperBounds[5] or transform[2, 3] < lowerBounds[5]:
                print 'Z translation incorrect'

            icpT.append(t)
            icpTransformsErr.append(err)
            icpTransforms.append(transform)

            # check to see if end error is worse then start error, if so then

            if visualize and False:  # display err plot of ICP
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(t, err, 'x--')
                ax.set_xlabel('Time')
                ax.set_ylabel('RMS error')
                ax.set_title('Result of ICP on %s Joint, initial transform %d' % (jointOfInterest.name, i))
                plt.show()

        icpTransformsErr = numpy.array(icpTransformsErr)

#        tindx = numpy.where(icpTransformsErr==icpTransformsErr.min())[0][0]
        tindx = numpy.where(icpTransformsErr == icpTransformsErr[:, -1].min())[0][0]
        transform = icpTransforms[tindx] * initialTransforms[tindx]
        print 'Best iteration was %d with transform:' % (tindx)
        print transform

        if visualize:  # display err plot of ICP
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(icpT[tindx], icpTransformsErr[tindx], 'x--')
            ax.set_xlabel('Time')
            ax.set_ylabel('RMS error')
            ax.set_title('Result of ICP on %s Joint, initial transform %d' % (jointOfInterest.name, tindx))
            plt.show()

        if visualize and False:
            # Display initial filtered points clouds that will be passed to ICP
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed = numpyTransform.transformPoints(transform, dataPoints)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Initial Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

            # display point clouds at final position
            modelPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            modelPoints = numpyTransform.transformPoints(cummulativeJointTransform, modelPoints)
            tri = numpy.array(range(modelPoints.shape[0] - modelPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(modelPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Model Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[1.0, 0.0, 0.0])

            dataPoints += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPoints = numpyTransform.transformPoints(cummulativeJointTransform, dataPoints)
            tri = numpy.array(range(dataPoints.shape[0] - dataPoints.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPoints, tri, scanData.isosurfaceJoint.parentJoint, name='Final Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 1.0, 0.0])

            dataPointsTransformed += jointOfInterest.initialLocationMat[:3, 3].getA().squeeze()
            dataPointsTransformed = numpyTransform.transformPoints(cummulativeJointTransform, dataPointsTransformed)
            tri = numpy.array(range(dataPointsTransformed.shape[0] - dataPointsTransformed.shape[0] % 3))
            tri = tri.reshape((tri.shape[0] / 3, 3))
            TriModel(dataPointsTransformed, tri, scanData.isosurfaceJoint.parentJoint, name='Final Translated Filtered Data Point Cloud for %s Joint' % (jointOfInterest.name), displayAsPoints=True, color=[0.0, 0.0, 1.0])

    else:  # skip ICP, just use predetermined transform
        transform = numpy.matrix(fineAlignTransforms['Hindlimb Right Ankle'])
#        transform = numpy.matrix(
#            [[  0.42756724,   0.53466485,   0.72891683, -28.76283195],
#             [ -0.88905172,   0.39466056,   0.23201311, -13.92154962],
#             [ -0.16362547,  -0.74724597,   0.64408863,  22.98018009],
#             [  0.,           0.,           0.,           1.        ]])

    # store transform
    fineAlignTransforms[jointOfInterest.name] = transform

    # Step 5: Apply ICP transform to joint
    R = numpy.matrix(numpy.identity(4))
    R[:3, :3] = transform[:3, :3]
    jointOfInterest.rotate(R, relative=True)
    cummulativeJointRotation = numpy.matrix(numpy.identity(4))
    cummulativeJointRotation[:3, :3] = cummulativeJointTransform[:3, :3]
    modifiedTranslate = cummulativeJointRotation * numpyTransform.translation(transform[:3, 3].getA().squeeze())  # translation has to include rotation effects of cumulative transformations
#    jointOfInterest.translate(modifiedTranslate[:3,3].getA().squeeze(), absolute=False)
    jointOfInterest.translate(transform[:3, 3].getA().squeeze(), absolute=False)

    if 'updateSceneFunc' in kwargs and kwargs['updateSceneFunc'] is not None:
        kwargs['updateSceneFunc']()
        time.sleep(3)

    # print resulting joint location
    if 'getResults' in kwargs and kwargs['getResults']:
        atlasData.atlasJoint.transformVertices()
        print '%s Joint Location: %s' % (jointOfInterest.name, str(jointOfInterest.location))
        meanDist = []
        for bone in bones:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            meanDist.append(numpy.sum(d) / len(i))
        print '%s Joint Mean surface to bone distance: %f' % (jointOfInterest.name, numpy.mean(meanDist))
        dist = 0.0
        numPoints = 0.0
        for bone in scanData.bonesVol1:
            d, i = scanData.isosurfaceModel1.kdtree.query(atlasData.atlasModels[bone].transformedVertexList)
            dist += numpy.sum(d)
            numPoints += len(i)
        print 'Post %s Align Mean distance all bones to surface: %f' % (jointOfInterest.name, dist / numPoints)

    # get indices of model vertices within 1mm of aligned atlas hip left vertices
    if 'lowerRightPawVertexShortcut' in kwargs and os.path.exists(kwargs['lowerRightPawVertexShortcut']):
        vertexDict = loadmat(kwargs['lowerRightPawVertexShortcut'])
        lowerRightPawindx = vertexDict['indx'].astype(numpy.bool).squeeze()
    else:
        # create KDTree of newly aligned bone
        dataPoints = []
        for bone in bones:
            atlasData.atlasJoint.transformVertices(modelID=id(atlasData.atlasModels[bone]))
            dataPoints = numpy.append(dataPoints, atlasData.atlasModels[bone].transformedVertexList)
        dataPoints = dataPoints.reshape((dataPoints.shape[0] / 3, 3))
        boneKDTree = KDTree(dataPoints)
        # find spine indices
        t1 = time.time()
        lowerRightPawindx = numpy.zeros(iso1KDTree.n, dtype=numpy.bool)
        NN = boneKDTree.query_ball_tree(iso1KDTree, 1.0 / scanData.sliceThickness)
        for vindx in NN:
            lowerRightPawindx[vindx] = True
        print 'Took %f seconds to calculate %s vertices' % (time.time() - t1, bone)
        if 'savePrefix' in kwargs and kwargs['savePrefix'] is not None:
            savemat(kwargs['savePrefix'] + 'lowerRightPawvertices.mat', {'indx': lowerRightPawindx})

    # save transforms to file
    if 'savePrefix' in kwargs:
        savemat(kwargs['savePrefix'] + 'FineAlignTransforms.mat', fineAlignTransforms)


def testGetSpine():
    data = loadmat('spineSeg.mat')
    getSpine(data['largestVolLabel'], data['neckPosAligned'])

if __name__ == '__main__':
#    alignmentAxis = RoughAlignTest(verbose=True, visualize=True)
#    print 'Rough Align Results'
#    print 'Alignment Axis:'
#    print alignmentAxis[0]
#    print 'Neck Position:'
#    print alignmentAxis[1]

#    savedData=loadmat('Saved Data.mat')
#    affineMatrixTest(savedData['referenceVolume'], savedData['axes'], alignedVolumeGoal=savedData['alignedCTVolume'])

#    roughAlignProfile()

    testGetSpine()
