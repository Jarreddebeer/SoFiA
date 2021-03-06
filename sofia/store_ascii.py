#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from numpy import *
import numpy as np


def make_ascii(objects, store_pars, outname,compress):
    print 'Store the results to ascii file: ', outname
    f1 = open(outname, 'w+')
    f1.write('# SoFia catalogue\n')
    f1.write('#\n')

    # write the parameters in
    sources = objects.getSources()
    pars = sources.itervalues().next().getParameters()
    f1.write('# ')
    if store_pars == ['*']:
        for i in sorted(pars):
            f1.write(i + '\t')
    else:
        for i in store_pars:
            f1.write(i + '\t')
    f1.write('\n')
    f1.write('#\n')

    # write the data....
    for i in sources:
        source_dict = sources[i].getParameters()
        if store_pars == ['*']:
            for j in sorted(source_dict):
                f1.write(str(source_dict[j].value) + '\t')
            f1.write('\n')
        else:

            for j in store_pars:
                f1.write(str(source_dict[j].value) + '\t')
            f1.write('\n')

    f1.close

    return


def make_ascii_from_array(
    objects,
    cathead,
    catunits,
    catfmt,
    store_pars,
    outname,
    compress
    ):
    print 'Store the results to ascii file: ', outname
    header = 'SoFia catalogue\n'

    objects = np.array(objects)

    # search for formatting with variable length
    lenCathead = []

    # print catfmt
    for j in catfmt:
        lenCathead.append(
            int(j.split('%')[1].split('e')[0].split('f')[0].split('i')[0].split('d')[0].split('.')[0]) + 1
            )

    # lenCathead[0] -= 2
    catNum = tuple(['(%i)' % jj for jj in range(len(cathead))])

    # creating the header
    header1 = ''
    header2 = ''
    header3 = ''
    if store_pars == ['*']:
        for i in range(0, len(cathead)):
            header1 += cathead[i].rjust(lenCathead[i])
            header2 += catunits[i].rjust(lenCathead[i])
            header3 += catNum[i].rjust(lenCathead[i])
        header += header1[3:] + '\n' + header2[3:] + '\n' + header3[3:]
    else:
        for i in range(0, len(store_pars)):
            if (store_pars[i] in cathead):
                index = list(cathead).index(store_pars[i])
                header1 += store_pars[i].rjust(lenCathead[index])
                header2 += catunits[index].rjust(lenCathead[index])
                header3 += catNum[index].rjust(lenCathead[index])
            else:
                sys.stderr.write("WARNING: Skipping undefined parameter \'" + str(store_pars[i]) + "\'.\n");
        header += header1[3:] + '\n' + header2[3:] + '\n' + header3[3:]

    if store_pars == ['*']:
        outputFormat = ''
        for i in range(0, len(catfmt)):
            outputFormat += catfmt[i] + ' '
        if compress:
	  outname += '.gz'
        np.savetxt(outname, np.array(objects), fmt=outputFormat, header=header)
    else:

        # copy all relevant parameters to a new array
        outputFormat = ''
        for par in store_pars:
            if (par in cathead): outputFormat += catfmt[list(cathead).index(par)] + ' '
        tmpObjects = []
        for obj in objects:
            tmpObjects.append([])
            for par in store_pars:
                if (par in cathead):
                    index = list(cathead).index(par)
                    tmpObjects[-1].append(obj[index])
        if compress:
	  outname += '.gz'
        np.savetxt(outname, np.array(tmpObjects), fmt=outputFormat, header=header)


