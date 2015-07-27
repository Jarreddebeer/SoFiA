#! /usr/bin/env python

# import default python libraries
import astropy.io.fits as astropy
import os
import sys
from numpy import *
import re
import imp


def read_data(doSubcube, inFile, weightsFile, maskFile, weightsFunction = None, subcube=[], subcubeMode='pixel'):
    # import the fits file into an numpy array for the cube and a dictionary for the header:
    # the data cube is converted into a 3D array
    if doSubcube:
        if os.path.isfile(inFile) == False:
            sys.stderr.write("ERROR: The specified data cube does not exist.\n")
            sys.stderr.write("       Cannot find: " + inFile + "\n")
            raise SystemExit(1)

        if len(subcube):
            try:
                imp.find_module('astropy')
                found = True
            except ImportError: found = False
            if found:
                from astropy import wcs
                from astropy.io import fits
                hdulist = fits.open(inFile, memmap=False)
                header = hdulist[0].header
                hdulist.close()

        if (len(subcube)==6 or len(subcube)==4) and subcubeMode=='world':
            print 'Calculating subcube boundaries from input WCS centre and radius'
            wcsin = wcs.WCS(header)
            # calculate cos(Dec) correction for RA range
            if wcsin.wcs.cunit[0]=='deg' and wcsin.wcs.cunit[1]=='deg': corrfact=cos(subcube[1]/180*pi)
            if header['naxis']==4: subcube=wcsin.wcs_world2pix(array([[subcube[0]-subcube[3]/corrfact,subcube[1]-subcube[4],subcube[2]-subcube[5],0],[subcube[0]+subcube[3]/corrfact,subcube[1]+subcube[4],subcube[2]+subcube[5],0]]),0)[:,:3]
            elif header['naxis']==3: subcube=wcsin.wcs_world2pix(array([[subcube[0]-subcube[3]/corrfact,subcube[1]-subcube[4],subcube[2]-subcube[5]],[subcube[0]+subcube[3]/corrfact,subcube[1]+subcube[4],subcube[2]+subcube[5]]]),0)
            elif header['naxis']==2: subcube=wcsin.wcs_world2pix(array([[subcube[0]-subcube[2]/corrfact,subcube[1]-subcube[3]],[subcube[0]+subcube[2]/corrfact,subcube[1]+subcube[3]]]),0)
            subcube=ravel(subcube,order='F')
            # make sure min pix coord is < max pix coord for all axes
            # this operation is meaningful because wcs_world2pix returns negative pixel coordinates only for pixels located before an axis' start
            # (i.e., negative pixel coordinates should not be interpreted as counting backward from an axis' end)
            if subcube[0]>subcube[1]: subcube[0],subcube[1]=subcube[1],subcube[0]
            if subcube[2]>subcube[3]: subcube[2],subcube[3]=subcube[3],subcube[2]
            if len(subcube)==6:
                if subcube[4]>subcube[5]: subcube[4],subcube[5]=subcube[5],subcube[4]
            # constrain subcube to be within the cube boundaries; if this is not possible then exit
            for ss in range(min(3,header['naxis'])):
                if ceil(subcube[1+2*ss])<0 or floor(subcube[2*ss])>=header['naxis%i'%(ss+1)]:
                    sys.stderr.write("ERROR: The requested subcube is outside the input cube along axis %i \n"%(ss))
                    raise SystemExit(1)
                else:
                    subcube[2*ss]=max(0,floor(subcube[2*ss]))
                    subcube[1+2*ss]=min(header['naxis%i'%(ss+1)]-1,ceil(subcube[1+2*ss]))+1
            subcube=list(subcube.astype(int))
        elif (len(subcube)==6 or len(subcube)==4) and subcubeMode=='pixel':
            # make sure pixel coordinates are integers
            for ss in subcube:
                if type(ss)!=int:
                    sys.stderr.write("ERROR: When subcubeMode = pixel the subcube must be defined by a set of integer pixel coordinates.\n")
                    sys.stderr.write("       The %i-th coordinate has the wrong type.\n" % subcube.index(ss))
                    raise SystemExit(1)
            # make sure to be within the cube boundaries
            for ss in range(min(3,header['naxis'])): subcube[2*ss]=max(0,subcube[2*ss])
            for ss in range(min(3,header['naxis'])): subcube[1+2*ss]=min(header['naxis%i'%(ss+1)],subcube[1+2*ss])
        elif len(subcube):
            sys.stderr.write("ERROR: import.subcubeMode can only be \'pixel\' or \'world\', and import.subcube must have 4 or 6 entries.\n")
            raise SystemExit(1)

        if len(subcube)==4: print 'Loading subcube of %s defined by [x1 x2 y1 y2] ='%inFile,subcube
        elif len(subcube)==6: print 'Loading subcube of %s defined by [x1 x2 y1 y2 z1 z2] ='%inFile,subcube
    else:
        print 'Loading cube: ' , inFile
        subcube = []
    f = astropy.open(inFile,memmap=False,do_not_scale_image_data=True)
    dict_Header = f[0].header

    # check whether the number of dimensions is acceptable and read data accordingly
    print
    # the default is three axis
    if dict_Header['NAXIS'] == 3:
        if len(subcube)==6:
            np_Cube = f[0].section[subcube[4]:subcube[5],subcube[2]:subcube[3],subcube[0]:subcube[1]]
            dict_Header['crpix1']-=subcube[0]
            dict_Header['crpix2']-=subcube[2]
            dict_Header['crpix3']-=subcube[4]
        elif not len(subcube): np_Cube = f[0].data
        else:
            sys.stderr.write("ERROR: The subcube list must have 6 entries (%i given).\n" % len(subcube))
            raise SystemExit(1)
    elif dict_Header['NAXIS'] == 4:
        if dict_Header['NAXIS4'] != 1:
            sys.stderr.write("ERROR: The size of the 4th dimension is > 1.\n")
            raise SystemExit(1)
        else:
            if len(subcube)==6:
                np_Cube = f[0].section[0,subcube[4]:subcube[5],subcube[2]:subcube[3],subcube[0]:subcube[1]]
                dict_Header['crpix1']-=subcube[0]
                dict_Header['crpix2']-=subcube[2]
                dict_Header['crpix3']-=subcube[4]
            elif not len(subcube): np_Cube = f[0].section[0]
            else:
                sys.stderr.write("ERROR: The subcube list must have 6 entries (%i given). Ignore 4th axis.\n" % len(subcube))
                raise SystemExit(1)
    elif dict_Header['NAXIS'] == 2:
        sys.stderr.write("WARNING: The input cube has 2 axes, third axis added.\n")
        print 'type: ', dict_Header['CTYPE1'], dict_Header['CTYPE2']
        print 'dimensions: ', dict_Header['NAXIS1'], dict_Header['NAXIS2']
        if len(subcube)==4:
            np_Cube = array([f[0].section[subcube[2]:subcube[3],subcube[0]:subcube[1]]])
            dict_Header['crpix1']-=subcube[0]
            dict_Header['crpix2']-=subcube[2]
        elif not len(subcube): np_Cube = array([f[0].data])
        else:
            sys.stderr.write("ERROR: The subcube list must have 4 entries (%i given).\n" % len(subcube))
            raise SystemExit(1)
    elif dict_Header['NAXIS'] == 1:
        sys.stderr.write("ERROR: The input has 1 axis, this is probably a spectrum instead of an 2D/3D image.\n")
        sys.stderr.write("       Type: " + str(dict_Header['CTYPE1']) + "\n")
        sys.stderr.write("       Dimensions: " + str(dict_Header['NAXIS1']) + "\n")
        raise SystemExit(1)
    else:
        sys.stderr.write("ERROR: The file has less than 1 or more than 4 dimensions.\n")
        raise SystemExit(1)

    f.close()

    if 'bscale' in dict_Header and 'bzero' in dict_Header:
        np_Cube*=dict_Header['bscale']
        np_Cube+=dict_Header['bzero']
        # NOTE: the above lines are more memory efficient than
        #np_Cube=np_Cube*dict_Header['bscale']+dict_Header['bzero']


    # check whether the axis are in the expected order:
    #if dict_Header['ctype1'][0:2] != 'RA' or dict_Header['ctype2'][0:3] != 'DEC':
    #	sys.stderr.write("WARNING: The dimensions are not in the expected order.\n")

    # the data cube has been loaded

    # apply weights if a weights file exists:
    if weightsFile:
        # check whether the weights cube exists:
        if os.path.isfile(weightsFile) == False:
            sys.stderr.write("ERROR: The defined weights cube does not exist.\n")
            sys.stderr.write("       Cannot find: " + weightsFile + "\n")
            raise SystemExit(1)

        else:
            # Scale the input cube with a weights cube
            # load the weights cube and convert it into a 3D array to be applied to the data 3D array
            # (note that the data has been converted into a 3D array above)
            print 'Loading and applying weights cube:', weightsFile
            f=astropy.open(weightsFile, memmap=False)
            dict_Weights_header=f[0].header
            if dict_Weights_header['NAXIS']==3:
                if len(subcube)==6: np_Cube*=f[0].section[subcube[4]:subcube[5],subcube[2]:subcube[3],subcube[0]:subcube[1]]
                else: np_Cube*=f[0].data
            elif dict_Weights_header['NAXIS']==4:
                if dict_Weights_header['NAXIS4']!=1:
                    sys.stderr.write("ERROR: The 4th dimension has more than 1 value.\n")
                    raise SystemExit(1)
                else:
                    sys.stderr.write("WARNING: The weights cube has 4 axes; first axis ignored.\n")
                    if len(subcube)==6: np_Cube*=f[0].section[0,subcube[4]:subcube[5],subcube[2]:subcube[3],subcube[0]:subcube[1]]
                    else: np_Cube*=f[0].section[0]
            elif dict_Weights_header['NAXIS']==2:
                sys.stderr.write("WARNING: The weights cube has 2 axes; third axis added.\n")
                if len(subcube)==6 or len(subcube)==4: np_Cube*=array([f[0].section[subcube[2]:subcube[3],subcube[0]:subcube[1]]])
                else: np_Cube*=array([f[0].data])
            elif dict_Weights_header['NAXIS']==1:
                sys.stderr.write("WARNING: The weights cube has 1 axis; interpreted as third axis; first and second axes added.\n")
                if len(subcube)==6: np_Cube*=reshape(f[0].section[subcube[4]:subcube[5]],(-1,1,1))
                elif not len(subcube): np_Cube*=reshape(f[0].data,(-1,1,1))
                else:
                    sys.stderr.write("ERROR: The subcube list must have 6 entries (%i given).\n" % len(subcube))
                    raise SystemExit(1)
            else:
                sys.stderr.write("ERROR: The weights cube has less than 1 or more than 4 dimensions.\n")
                raise SystemExit(1)

            f.close()
            print 'Weights cube loaded and applied.'

    elif weightsFunction:
        # WARNING: I'm not sure if there is a safe way to properly implement multiplication of a data array
        # WARNING: with a user-specified function in Python without the need for a whitelist, nested loops,
        # WARNING: or the creation of multiple copies of the cube.
        print "Evaluating function: %s"%weightsFunction

        # Define whitelist of allowed character sequences:
        whitelist = ["x", "y", "z", "e", "E", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "exp", "log", "sqrt", "square", "power", "absolute", "fabs", "sign"]

        # Search for all keywords consisting of consecutive sequences of alphabetical characters:
        # NOTE: Explicit conversion to string is required unless readoptions.py is modified!
        keywordsFound = filter(None, re.split("[^a-zA-Z]+", str(weightsFunction)))

        # Check for non-whitelisted sequences:
        for keyword in keywordsFound:
            if keyword not in whitelist:
                sys.stderr.write("ERROR: Unsupported keyword/function found in weights function:\n")
                sys.stderr.write("         %s\n" % weightsFunction)
                sys.stderr.write("       Please check your input.\n")
                raise SystemExit(1)

        # hardcoded number of weights chunks
        Nz=50
        # check that the number of weights z-chunks is at most equal to the total nr of chans
        Nz=min(Nz,np_Cube.shape[0])
        # calculate the size of each chunk along the Z axis (rounding to integer)
        Lz=int(round(float(np_Cube.shape[0])/Nz))
        # calculate number of chunks really needed given above rounding
        Nz=int(ceil(float(np_Cube.shape[0])/Lz))
        print "Evaluating and applying weights function in %i chunks along the Z axis"%Nz
        for zz in range(Nz):
            # last chunk may have different length than the others
            if zz==Nz-1: z,y,x=indices((np_Cube.shape[0]-Lz*zz,np_Cube.shape[1],np_Cube.shape[2]))
            else: z,y,x=indices((Lz,np_Cube.shape[1],np_Cube.shape[2]))
            z+=zz*Lz
            try:
                # NOTE: eval() should be safe now as we don't allow for non-whitelisted sequences...
                np_Cube[z,y,x] *= eval(str(weightsFunction))
                # WARNING: There is no check here whether the expression to be evaluated is actually valid,
                #          e.g. SoFiA will crash if the weights function is sqrt(-1). 'try' doesn't catch this!
                #          Even if we set np.seterr(all='raise'), we still run into problems with expressions
                #          that are valid but not floating-point numbers, e.g. sqrt((1,2)).
            except:
                sys.stderr.write("ERROR: Failed to evaluate weights function:\n")
                sys.stderr.write("         %s\n" % weightsFunction)
                sys.stderr.write("       Please check your input.\n")
                raise SystemExit(1)
        print "Function-weighted cube created.\n"

    if maskFile:
        # check whether the mask cube exists:
        if os.path.isfile(maskFile) == False:
            sys.stderr.write("ERROR: The specified mask cube does not exist.\n")
            sys.stderr.write("       Cannot find: " + maskFile + "\n")
            #print 'WARNING: Program continues, without using input mask'
            #mask=zeros(np_Cube.shape)
            raise SystemExit(1)

        else:
            print 'Loading mask cube: ' , maskFile
            g = astropy.open(maskFile,memmap=False)
            mask = g[0].data
            mask[mask>0]=1
            g.close()
            print 'Mask cube loaded.'
    else: mask = zeros(np_Cube.shape, dtype=bool)

    # The original data is replaced with the Weighted cube!
    # If weighting is being used, the data should be read in again during parameterisation.
    return np_Cube, dict_Header, mask, subcube
