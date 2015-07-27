#! /usr/bin/env python

# import default python libraries
import sys, os
from time import time

# import source finding modules
sys.path.insert(0, os.environ['SOFIA_MODULE_PATH'])
import numpy as np
from sofia import functions
from sofia import readoptions
from sofia import import_data
from sofia import pyfind

default_file = '%s/SoFiA_default_input.txt'%(os.path.dirname(os.path.realpath(__file__)))
Parameters = readoptions.readPipelineOptions(default_file)

# ---------------------
# ---- IMPORT DATA ----
# ---------------------

t_read_start = time()
np_Cube, dict_Header, mask, subcube = import_data.read_data(Parameters['steps']['doSubcube'],**Parameters['import'])
t_read = time() - t_read_start
print 'Reading cube took: %.3f' % t_read

t_rms_start = time()
globalrms=functions.GetRMS(np_Cube,rmsMode='negative',zoomx=1,zoomy=1,zoomz=1,verbose=True)
t_rms = time() - t_rms_start
print 'Global RMS (%d) took: %.3f' % (t_rms, globalrms)

# --- PYFIND ---
t_sc_start = time()
if Parameters['steps']['doSCfind']:
    print 'Running S+C filter'
    pyfind_mask = pyfind.SCfinder_mem(np_Cube, dict_Header, **Parameters['SCfind'])
    np.save('tests/original/pyfind_mask', pyfind_mask)
    mask = mask + pyfind_mask
t_sc = time() - t_sc_start

print 'S+C took: %.3f' % t_sc
sys.stdout.flush()
