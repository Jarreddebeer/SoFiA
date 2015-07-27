#! /usr/bin/env python

import numpy as np
import math as mt
import scipy.ndimage as nd
from functions import GetRMS
from time import time

def GaussianNoise(F,N0,s0):
    return N0*np.exp(-F**2/2/s0**2)

def SizeFilter(mskt,sfx,sfy,sfz,sbx,sby,sbz,zt,sizeFilter,edgeMode='constant',verbose=0):
	mskt=nd.filters.gaussian_filter(mskt,[0,mt.sqrt(sfy**2+sby**2)/2.355,mt.sqrt(sfx**2+sbx**2)/2.355],mode=edgeMode)
	if zt=='b': mskt=nd.filters.uniform_filter1d(mskt,max(sbz,sfz),axis=0,mode=edgeMode)
	elif zt=='g': mskt=nd.filters.gaussian_filter1d(mskt,max(sbz,sfz/2.355),axis=0,mode=edgeMode)
	mskt[mskt< sizeFilter]=0
	mskt[mskt>=sizeFilter]=1
	return mskt

def MaskedCube(incube,msk,replace_value):
	maskedcube=np.copy(incube)
	maskedcube[msk]=np.sign(incube[msk])*np.minimum(abs(incube[msk]),replace_value)
	# this only decreases the absolute value of voxels already in the mask, or leaves it unchanged
	# if already lower than replace_value; the sign is unchanged
	return maskedcube

def SortKernels(kernels):
	# Sorting kernels
	uniquesky=[]
	velsmooth=[]
	velfshape=[]
	for jj in np.array(kernels):
		if list(jj[:2].astype(float)) not in uniquesky: uniquesky.append(list(jj[:2].astype(float)))
	uniquesky=[kk[1] for kk in sorted([(float(jj[0]),jj) for jj in uniquesky])]

	for jj in uniquesky:
		velsmooth.append([])
		velfshape.append([])
		for ii in np.array(kernels):
			if list(ii[:2].astype(float))==jj:
				velsmooth[-1].append(int(ii[2]))
				velfshape[-1].append(ii[3])
	return uniquesky,velsmooth,velfshape


def SCfinder_mem(cube,header,kernels=[[0,0,0,'b'],],threshold=3.5,sizeFilter=0,maskScaleXY=2.,maskScaleZ=2.,kernelUnit='pixel',edgeMode='constant',rmsMode='negative',verbose=0):

    msk = np.zeros(cube.shape, 'bool')
    found_nan = np.isnan(cube).sum()

    # Set dn x dn x dn box boundaries where to measure noise
    dn = 100
    n0 = max(0, int(float(cube.shape[0] - dn) / 2))
    n1 = max(0, int(float(cube.shape[1] - dn) / 2))
    n2 = max(0, int(float(cube.shape[2] - dn) / 2))

    # Measure noise in original (sub-) cube
    t_rms_initial_start = time()
    rms = GetRMS(
        cube[
            n0 : cube.shape[0] - n0,
            n1 : cube.shape[1] - n1,
            n2 : cube.shape[2] - n2
        ], rmsMode=rmsMode,zoomx=1,zoomy=1,zoomz=1,verbose=verbose
    )
    t_rms_initial = time() - t_rms_initial_start
    print 'initial RMS: %.3f' % t_rms_initial

    cube_to_smooth = cube * 1.
    if found_nan: cube_to_smooth = np.nan_to_num(cube_to_smooth)

    for kernel in kernels:
        [kx, ky, kz, kt] = kernel
        t_start = time()

        if kernelUnit == 'world' or kernelUnit == 'w':
            kx = abs(float(kx) / header['cdelt1'])
            ky = abs(float(ky) / header['cdelt2'])
            kz = abs(float(kz) / header['cdelt3'])

        if kt == 'b':
            kz_ceil = int(mt.ceil(kz))
            if kz != kz_ceil:
                kz = kz_ceil
                if verbose: print '    WARNING: Rounding width of boxcar z kernel to next integer'

        cube_smoothed = cube_to_smooth * 1.
        mask_positive = (cube_smoothed > 0) * msk
        mask_negative = (cube_smoothed < 0) * msk
        cube_smoothed[mask_positive] = +maskScaleXY * rms
        cube_smoothed[mask_negative] = -maskScaleXY * rms

        if kx + ky:
            cube_smoothed = nd.filters.gaussian_filter(cube_smoothed, [0, ky / 2.355, kx / 2.355], mode = edgeMode)

        if kz:
            if kt == 'b':
                cube_smoothed = nd.filters.uniform_filter1d(cube_smoothed, kz, axis = 0, mode = edgeMode)
            elif kt == 'g':
                cube_smoothed = nd.filters.gaussian_filter1d(cube_smoothed, kz / 2.355, axis = 0, mode = edgeMode)

        t_rms_start = time()

        rms_smoothed = GetRMS(
            cube_smoothed[
                n0 : cube.shape[0] - n0,
                n1 : cube.shape[1] - n1,
                n2 : cube.shape[2] - n2
            ],
            rmsMode = rmsMode, zoomx = 1, zoomy = 1, zoomz = 1, verbose = verbose
        )

        t_rms = time() - t_rms_start

        mask_threshold_positive = (cube_smoothed >=  threshold * rms_smoothed)
        mask_threshold_negative = (cube_smoothed <= -threshold * rms_smoothed)
        msk = msk + mask_threshold_positive + mask_threshold_negative

        filename = 'tests/original_refactored/%s-%s-%s-%s' % (kx, ky, kz, kt)
        np.save(filename, cube_smoothed)
        del(cube_smoothed)

        t_finder = time() - t_start
        print 'Filter %s %s %s %s: %.3fs      RMS: %.3fs' % (kx, ky, kz, kt, t_finder, t_rms)

    return msk
