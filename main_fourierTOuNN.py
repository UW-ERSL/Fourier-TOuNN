import numpy as np
import time
from TOuNN import TopologyOptimizer
import matplotlib.pyplot as plt

nelx = 60; # number of FE elements along X
nely = 30; # number of FE elements along Y
elemSize = np.array([1.0,1.0]);
mesh = {'nelx':nelx, 'nely':nely, 'elemSize':elemSize};

matProp = {'E':1.0, 'nu':0.3}; # Structural
matProp['K'] = 1.0; # Thermal
matProp['penal'] = 1; # SIMP penalization constant, starting value

exampleName = 'TipCantilever'
physics = 'Structural'
ndof = 2*(nelx+1)*(nely+1);
force = np.zeros((ndof,1))
dofs=np.arange(ndof);
fixed = dofs[0:2*(nely+1):1];
force[2*(nelx+1)*(nely+1)-2*nely+1, 0 ] = -1;
symXAxis = {'isOn':False, 'midPt':0.5*nely};
symYAxis = {'isOn':False, 'midPt':0.5*nelx};
bc = {'exampleName':exampleName, 'physics':physics, \
      'force':force, 'fixed':fixed, 'symXAxis':symXAxis, 'symYAxis':symYAxis };

# For more BCs see examples.py

nnSettings = {'numLayers':1, 'numNeuronsPerLyr':20 }

fourierMinRadius, fourierMaxRadius = 4, 10;
numTerms = 250;
fourierMap = {'isOn':True, 'minRadius':fourierMinRadius, \
              'maxRadius':fourierMaxRadius, 'numTerms':numTerms};
    
densityProjection = {'isOn':True, 'sharpness':8};
desiredVolumeFraction = 0.6;

minEpochs = 150; # minimum number of iterations
maxEpochs = 500; # Max number of iterations

plt.close('all');
overrideGPU = False
start = time.perf_counter()
topOpt = TopologyOptimizer(mesh, matProp, bc, nnSettings, fourierMap, \
                  desiredVolumeFraction, densityProjection, overrideGPU);
topOpt.optimizeDesign(maxEpochs,minEpochs);
print("Time taken (secs): {:.2F}".format( time.perf_counter() - start))