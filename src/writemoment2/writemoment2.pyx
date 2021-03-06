#!/usr/bin/env python
import astropy.io.fits as pyfits
import os
import numpy as np
cimport numpy as np
from libc.math cimport isnan
import sys


def removeOptions(dictionary):
  modDictionary = dictionary
  for key in modDictionary['steps']:
    if modDictionary['steps'][key] == 'False':
      modDictionary.pop(key.split('do')[1].lower(), None)
  return modDictionary


def recursion(dictionary,optionsList,optionsDepth,counter=0):
  if type(dictionary)==type({}):
    for k in dictionary:
      optionsList.append(str(k))
      optionsDepth.append(counter)
      recursion(dictionary[k],optionsList,optionsDepth,counter=counter+1)
  else:
    optionsList[len(optionsList)-1]+='='+str(dictionary)
    counter = 0


def regridMaskedChannels(datacube,maskcube,header):
  
  maskcubeFlt = maskcube.astype('float')
  maskcubeFlt[maskcubeFlt>1] = 1
  
  print 'Regridding...'
  sys.stdout.flush()
  from scipy import interpolate
  z=(np.arange(1.,header['naxis3']+1)-header['crpix3'])*header['cdelt3']+header['crval3']
  if header['ctype3']=='VELO-HEL':
    pixscale=(1-header['crval3']/2.99792458e+8)/(1-z/2.99792458e+8)
  else:
    sys.stderr.write("WARNING: Cannot convert axis3 coordinates to frequency. Will ignore the effect of CELLSCAL = 1/F.\n")
    pixscale=np.ones((header['naxis3']))
  x0,y0=header['crpix1']-1,header['crpix2']-1
  xs=np.arange(datacube.shape[2],dtype=float)-x0
  ys=np.arange(datacube.shape[1],dtype=float)-y0
  for zz in range(datacube.shape[0]):
    regrid_channel=interpolate.RectBivariateSpline(ys*pixscale[zz],xs*pixscale[zz],datacube[zz])
    datacube[zz]=regrid_channel(ys,xs)
    regrid_channel_mask=interpolate.RectBivariateSpline(ys*pixscale[zz],xs*pixscale[zz],maskcubeFlt[zz])
    maskcubeFlt[zz] = regrid_channel_mask(ys,xs)
  
  maskMin = maskcubeFlt.min()
  datacube[abs(maskcubeFlt)<=abs(maskMin)] = 0
  del maskcubeFlt
  
  return datacube

def writeMoments(datacube,maskcube,filename,debug,header,compress,domom0,domom1):
  nrdetchan=(maskcube>0).sum(axis=0)
  if nrdetchan.max()<65535: nrdetchan=nrdetchan.astype('int16')
  else: nrdetchan=nrdetchan.astype('int32')
  hdu = pyfits.PrimaryHDU(data=nrdetchan,header=header)
  hdu.header['bunit']='Nchan'
  hdu.header['datamin']=nrdetchan.min()
  hdu.header['datamax']=nrdetchan.max()
  del(hdu.header['crpix3'])
  del(hdu.header['crval3'])
  del(hdu.header['cdelt3'])
  del(hdu.header['ctype3'])
  name = '%s_nrch.fits'%filename
  if compress:
    name += '.gz'
  hdu.writeto(name,output_verify='warn',clobber=True)
  
  datacube[maskcube==0]=0
  if 'cellscal' in header:
    if header['cellscal'] == '1/F':
      print 'WARNING: CELLSCAL keyword with value 1/F found.'
      print 'Will regrid masked cube before making moment images.'
      datacube=regridMaskedChannels(datacube,maskcube,header)
  datacube = np.array(datacube, dtype=np.single)
  if domom0 or domom1:
    m0 = mom0(datacube)
  if domom0:
    print 'Writing moment-0' # in units of header['bunit']*km/s
    if 'vopt' in header['ctype3'].lower() or 'vrad' in header['ctype3'].lower() or 'velo' in header['ctype3'].lower() or 'felo' in header['ctype3'].lower():
      if not 'cunit3' in header or header['cunit3'].lower()=='m/s': dkms=abs(header['cdelt3'])/1e+3 # assuming m/s
      elif header['cunit3'].lower()=='km/s': dkms=abs(header['cdelt3'])
      bunitExt = '.km/s'
    elif 'freq' in header['ctype3'].lower():
      #if not 'cunit3' in header or header['cunit3'].lower()=='hz': dkms=abs(header['cdelt3'])/1.42040575177e+9*2.99792458e+5 # assuming Hz
      #elif header['cunit3'].lower()=='khz': dkms=abs(header['cdelt3'])/1.42040575177e+6*2.99792458e+5
      if not 'cunit3' in header or ('cunit3' in header and header['cunit3'].lower()=='hz'):
        bunitExt = '.Hz'
      else: 
        bunitExt = '.'+header['cunit3']
      dkms=1. # no scaling, avoids crashing
    hdu = pyfits.PrimaryHDU(data=m0*dkms,header=header)
    hdu.header['bunit']+=bunitExt
    hdu.header['datamin']=(m0*dkms).min()
    hdu.header['datamax']=(m0*dkms).max()
    del(hdu.header['crpix3'])
    del(hdu.header['crval3'])
    del(hdu.header['cdelt3'])
    del(hdu.header['ctype3'])
    hdu.header['cellscal']='constant'
    if debug: hdu.writeto('%s_mom0.debug.fits'%filename,output_verify='warn',clobber=True)
    else: 
      name = '%s_mom0.fits'%filename
      if compress:
        name += '.gz'
      hdu.writeto(name,output_verify='warn',clobber=True)
  if domom1:
    print 'Writing moment-1'
    m1 = mom1(datacube,m0,header['crpix3'],header['crval3'],header['cdelt3'])
    # convert it to km/s (using radio velocity definition to go from Hz to km/s)
    if 'vopt' in header['ctype3'].lower() or 'vrad' in header['ctype3'].lower() or 'velo' in header['ctype3'].lower() or 'felo' in header['ctype3'].lower():
      if not 'cunit3' in header:
        m1/=1e+3 # assuming m/s
        bunitExt = 'km/s'
      elif header['cunit3'].lower()=='km/s': 
        bunitExt = 'km/s'
      else:
        bunitExt = header['cunit3']
    elif 'freq' in header['ctype3'].lower():
      #if not 'cunit3' in header or header['cunit3'].lower()=='hz': m1*=2.99792458e+5/1.42040575177e+9 # assuming Hz
      #elif header['cunit3'].lower()=='khz': m1*=2.99792458e+5/1.42040575177e+6
      if not 'cunit3' in header or ('cunit3' in header and header['cunit3'].lower()=='hz'):
        bunitExt = 'Hz'
      else: 
        bunitExt = header['cunit3']
      dkms=1. # no scaling, avoids crashing
    # calculate moment 1
    hdu = pyfits.PrimaryHDU(data=m1,header=header)
    hdu.header['bunit']=bunitExt
    hdu.header['datamin']=np.nanmin(m1)
    hdu.header['datamax']=np.nanmax(m1)
    del(hdu.header['crpix3'])
    del(hdu.header['crval3'])
    del(hdu.header['cdelt3'])
    del(hdu.header['ctype3'])
    hdu.header['cellscal']='constant'
    if debug: hdu.writeto('%s_mom1.debug.fits'%filename,output_verify='warn',clobber=True)
    else: 
      name = '%s_mom1.fits'%filename
      if compress:
        name += '.gz'
      hdu.writeto(name,output_verify='warn',clobber=True)


def mom0(cube1):
  
  cdef:
    int i, j, k
    double sum
    double [:,:] mom0 = np.zeros((cube1.shape[1],cube1.shape[2]))
    float[:,:,:] cube = cube1
    
  for j in range(cube.shape[1]):
    for k in range(cube.shape[2]):
      sum = 0
      for i in range(cube.shape[0]):
        if not isnan(cube[i,j,k]) and cube[i,j,k]!=0:
          sum += cube[i,j,k]
      mom0[j,k] = sum

  return np.array(mom0)

def mom1(cube1, cube2, int cpx, float cval, float cdelt):

  cdef:
    int i, j, k
    double sum
    double [:,:] mom1 = np.zeros((cube1.shape[1],cube1.shape[2]))
    float[:,:,:] cube = cube1
    double[:,:] mom0 = cube2
  
  for j in range(cube.shape[1]):
    for k in range(cube.shape[2]):
      sum = 0
      for i in range(cube.shape[0]):
        if not isnan(cube[i,j,k]) and cube[i,j,k]!=0:
          sum += ((i+1-cpx)*cdelt+cval)*cube[i,j,k]
      if mom0[j,k] != 0 and not isnan(mom0[j,k]):
        mom1[j,k] = sum/mom0[j,k]
      else:
        mom1[j,k] = np.nan

  return np.array(mom1)
