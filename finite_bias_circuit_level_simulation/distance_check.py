#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
from pathlib import Path
from codedistance import *

if len(sys.argv) > 1:
    CircuitFile = sys.argv[1]
else:
    # Default to the CSS example circuit bundled with the repo.
    # Run: python distance_check.py <path/to/circuit.stim> to check any circuit.
    _here = Path(__file__).resolve().parent
    CircuitFile = str(_here / "distance_check" / "circuit_css_l6_m6.stim")

qc = stim.Circuit.from_file(CircuitFile)

params = {
    'iterCount': 100,     ## number of iterations - eg QDistRnd, decoderDist
    'maxTime': 3600*8,      ## max runtime for CLI methods (eg SAT, m4ri, )
    'nThreads': 1,
    'LOCheck': 1           ## cater for multi-threaded methods
}

## Magma distance finding: requires Magma licence and magma executable in PATH
# method = 'magmaMinWeight'
# method = 'magmaMinWord'
# method = 'magmaWEDist'

## Serban Cercelescu C Library https://github.com/qubitserfed/Qubitserf
## requires compiled  qubitserf executable in PATH
## multi-threaded option available
## Brouwer-Zimmermann
# method = 'qubitSerfBZ'
## Meet in the middle
# method = 'qubitSerfMM'

## Stim methods - truncated Tanner graph traversal
## requires stim pypi package
# method='UndetectableErrorStim'
# method = 'GraphLikeErrorStim'
# method='ColourCodeDistStim'

## MW implementations of Stim methods
# method='UndetectableErrorMW'
# method='GraphLikeErrorMW'
# method='ColourCodeDistMW'

## MW implementation of QDistRnd
# method = 'QDistRndMW'

## Pryadko dist m4ri C package - requires dist_m4ri binary in $PATH
# method = 'dist_m4ri_RW'
# method = 'dist_m4ri_CC'

## SAT solvers:
## requires pysat python library
# method = 'pySATDist'
## Command line interface - requires a SAT solver binary to be in the $PATH - default is cashwmaxsatcoreplus
# method = 'CLISATDist'

## Gurobi MIP solver - requires gurobipy python package
# method = 'GurobiDist'

## OR-tools MIP solver - requires ortools python package
# method = 'MIPDist' 
## various MIP solvers can be used by OR-tools
params['MIP_solver'] = 'SCIP' 
##     CBC_MIXED_INTEGER_PROGRAMMING or CBC
##     SAT_INTEGER_PROGRAMMING or SAT or CP_SAT
##     SCIP_MIXED_INTEGER_PROGRAMMING or SCIP
##     GUROBI_MIXED_INTEGER_PROGRAMMING or GUROBI or GUROBI_MIP - requires full installation of Gurobi - https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer
##     CPLEX_MIXED_INTEGER_PROGRAMMING or CPLEX or CPLEX_MIP - requires installation of CPLEX https://www.ibm.com/docs/en/icos
##     XPRESS_MIXED_INTEGER_PROGRAMMING or XPRESS or XPRESS_MIP - requires installation of Xpress https://www.fico.com/fico-xpress-optimization/docs/latest/getting_started/dhtml/chap16.html
##     GLPK_MIXED_INTEGER_PROGRAMMING or GLPK or GLPK_MIP - requires installation of GLPK: https://www.gnu.org/software/glpk/''',type=str, default='SCIP')

## distance-finding via syndrom decoding
method = 'decoderDist'
## Type of decoder - for decoderDist only. Options are bposd, bplsd, sebplsd, tesseract, tesseractLongbeam",type=str, default='bposd')
params['decoder'] = 'bposd' 
## method=='decoderDist' and decoder in {'bposd','bplsd','sebplsd'} only
params["bp_method"] = 'product_sum' ## Belief Propagation method
params["bp_schedule"] = 'parallel' ## Belief Propagation schedule type - options parallel or serial
params["bp_max_iter"] = 100  ## Belief Propagation max iterations
params["osd_method"] = 'OSD_CS' 
params["osd_order"] = 1 


# allMethods = [
#               'UndetectableErrorStim',
#               'GraphLikeErrorStim',
#               'ColourCodeDistStim',
#               'UndetectableErrorMW',
#               'GraphLikeErrorMW',
#               'ColourCodeDistMW',
#               'dist_m4ri_RW',
#               'dist_m4ri_CC',
#               'QDistRndMW',
#               'decoderDist',
#               'GurobiDist',
#               'MIPDist',
#               'pySATDist'
#               'magmaMinWeight',
#               'magmaMinWord',
#               'magmaWEDist',
#               'qubitSerfBZ',
#               'qubitSerfMM',
#               ]

allMethods = ['QDistEvol']

## Single Method
methods = [method]
## All methods 
methods = allMethods

startTimer()
for method in methods:
    print(method)
    res = circuitDistance(qc,method=method,params=params,seed=0)
    print(f"Calculated distance: {res['d']}")
    print(f"Time Taken: {elapsedTime()}")
    if (res['T'] > 1):
        print(f"Total Trials: {res['T']}")
        print(f"Trials at d={res['d']}: {res['R']}")
    lo = res['L']
    if (np.sum(lo)) > 0:
        # r1,V = HowRes(Hz,lo,2)
        print(f'lo: {bin2Set(lo)}')
        # lo = ZMat2D(lo)
        # r2 = matMul(lo,Hx.T,2)
        # print(f'Logical Operator Check: commutes with stabilsers {np.sum(r2)==0}; non-trivial {np.sum(r1)!=0}')
    # print(res['progress'])
    print("###########################################\n")


# In[ ]:




