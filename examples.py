import numpy as np
import time
from TOuNN import TopologyOptimizer
import matplotlib.pyplot as plt
#  ~~~~~~~~~~~~ Setup ~~~~~~~~~~~~~#
example = 1; # see below for description
#  ~~~~~~~~~~~~Mesh sizes~~~~~~~~~~~~~#
nelx = 60; # number of FE elements along X
nely = 30; # number of FE elements along Y
elemSize = np.array([1.0,1.0]);
penal = 1; # SIMP penalization constant, starting value
#  ~~~~~~~~~~~~NN Size~~~~~~~~~~~~~#
numLayers = 1; # the depth of the NN
numNeuronsPerLyr = 20; # the height of the NN
#  ~~~~~~~~Optimization Params~~~~~~~~~~~#
minEpochs = 150; # minimum number of iterations
maxEpochs = 500; # Max number of iterations
useSavedNet = False;# use a net previouslySaved  as starting point (exampleName_nelx_nely.nt in ./results folder)
#  ~~~~~~~Fourier params~~~~~~~~~~~#
fourierMinRadius, fourierMaxRadius = 4, 10;
numTerms = 250;
fourierMap = {'isOn':True, 'minRadius':fourierMinRadius, 'maxRadius':fourierMaxRadius, 'numTerms':numTerms};
#  ~~~~~~~Experimental~~~~~~~~~~~#
localVolumeControl = {'isOn':False, 'radius':5};
densityProjection = {'isOn':False, 'sharpness':16};
maxLengthScaleControl = {'isOn':False, 'radius':9, 'voidVol':0.05*np.pi*9**2};
#  ~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~#
if(example == 1): # tip cantilever
    exampleName = 'TipCantilever'
    physics = 'Structural'
    meshType = 'rectGeom';
    matProp = {'E':1.0, 'nu':0.3};
    desiredVolumeFraction = 0.6; # between 0.1 and 0.9 
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed = dofs[0:2*(nely+1):1];
    force[2*(nelx+1)*(nely+1)-2*nely+1, 0 ] = -1;
    nonDesignRegion = None;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 2): # mid cantilever
    exampleName = 'MidCantilever'
    physics = 'Structural';
    meshType = 'rectGeom';
    matProp = {'E':1.0, 'nu':0.3};
    desiredVolumeFraction = 0.5; # between 0.1 and 0.9 
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed = dofs[0:2*(nely+1):1];
    force[2*(nelx+1)*(nely+1)- (nely+1), 0 ] = -1;
    nonDesignRegion = None;
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 3): #  MBBBeam
    desiredVolumeFraction = 0.5; # between 0.1 and 0.9 
    exampleName = 'MBBBeam'
    physics = 'Structural'
    meshType = 'rectGeom';
    matProp = {'E':1.0, 'nu':0.3};
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed= np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1);
    force[2*(nely+1)+1 ,0]=-1;
    nonDesignRegion = None;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 4): #  Michell
    desiredVolumeFraction = 0.4; # between 0.1 and 0.9 
    exampleName = 'Michell'
    physics = 'Structural'
    meshType = 'rectGeom';
    matProp = {'E':1.0, 'nu':0.3};
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed=np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely] ); # ,2*(nelx+1)*(nely+1)-2*nely+1,
    force[nelx*(nely+1)+2 ,0]=-1;
    nonDesignRegion = None;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':True, 'midPt':0.5*nelx};

elif(example == 5): #  DistributedMBB
    exampleName = 'Bridge'
    physics = 'Structural'
    meshType = 'rectGeom';
    desiredVolumeFraction = 0.45; # between 0.1 and 0.9 
    matProp = {'E':1.0, 'nu':0.3};
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed= np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1,2*(nelx+1)*(nely+1)-2*nely] );
    force[2*nely+1:2*(nelx+1)*(nely+1):8*(nely+1),0]=-1/(nelx+1);
    nonDesignRegion = None # s{'x>':0, 'x<':nelx,'y>':nely-1,'y<':nely}; # None #
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':True, 'midPt':0.5*nelx};

elif(example == 6): # Tensile bar
    exampleName = 'TensileBar'
    physics = 'Structural'
    meshType = 'rectGeom';
    nelx = 20; # number of FE elements along X
    nely = 10; # number of FE elements along Y
    numLayers = 1; # the depth of the NN
    numNeuronsPerLyr = 1; # the height of the NN
    desiredVolumeFraction = 0.4; # between 0.1 and 0.9 
    matProp = {'E':1.0, 'nu':0.3};
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof); 
    fixed =np.union1d(np.arange(0,2*(nely+1),2), 1); # fix X dof on left
    midDofX= 2*(nelx+1)*(nely+1)- (nely);
    force[midDofX, 0 ] = 1;
    nonDesignRegion = None;
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 7): #  two fixed ends
    desiredVolumeFraction = 0.675; # between 0.1 and 0.9 
    exampleName = 'twoFixedEnds'
    physics = 'Structural'
    meshType = 'rectGeom';
    matProp = {'E':1.0, 'nu':0.3};
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed= np.union1d(np.arange(0,2*(nely+1),1), np.arange(2*(nely+1)*(nelx),2*(nely+1)*(nelx+1),1));
    force[nelx*(nely+1)+1 ,0]=-1;
    nonDesignRegion = None;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':True, 'midPt':0.5*nelx};

elif(example == 8): # Tapered bar
    exampleName = 'TaperedBar'
    physics = 'Thermal'
    meshType = 'rectGeom';
    desiredVolumeFraction = 0.5; # between 0.1 and 0.9 
    matProp = {'K':1.0}; #{'E':1.0, 'nu':0.3};
    ndof = (nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof); 
    fixed =np.union1d(np.arange(0,(nely+1),1), 1); # fix dofs on left
    midDofX= (nelx+1)*(nely+1)- int(0.5*nely); # right edge mid DOF
    force[midDofX-4:midDofX+4:1, 0 ] = 1;
    nonDesignRegion = None;
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};
    
elif(example == 9):     
    exampleName = 'rightEdgeHeatLoad'
    physics = 'Thermal'
    meshType = 'rectGeom';
    matProp = {'K':1.0};
    desiredVolumeFraction = 0.3; # between 0.1 and 0.9 
    ndof = (nelx+1)*(nely+1);
    force = np.zeros((ndof,1));
    force[np.arange(int(nelx*(nely+1)) , int((nelx+1)*(nely+1)) )] = 1;
    fixed =  int(nely/2 + 1 - nely/20) # np.arange(0,nely+1,1)# 
    nonDesignRegion = None # {'x>':int(0.125*nelx), 'x<':int(0.625*nelx),'y>':int(0.125*nely),'y<':int(0.625*nely)};
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 10): # BodyLoad
    exampleName = 'bodyHeatLoad'
    physics = 'Thermal'
    meshType = 'rectGeom';
    matProp = {'K':1.0};
    desiredVolumeFraction = 0.5; # between 0.1 and 0.9 
    ndof = (nelx+1)*(nely+1);
    force = 0.*np.ones((ndof,1));
    force[0::1] = 2.5e-3
    # force = (2.5e-3)*np.ones((ndof,1));
    fixed =  np.arange(int(nely/2 -5) , int(nely/2 + 5) ); # int(nely/2 + 1 - nely/20) #
    nonDesignRegion = None # {'x>':int(0.125*nelx), 'x<':int(0.625*nelx),'y>':int(0.125*nely),'y<':int(0.625*nely)};
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

# ~~ Package the infos into a dictionary~~~~~~~~#       
meshFile = {'meshType':meshType, 'nelx':nelx, 'nely':nely, 'elemSize': elemSize, 'force':force, 'fixed':fixed};
#%%
plt.close('all');
overrideGPU = False
start = time.perf_counter()
topOpt = TopologyOptimizer();
topOpt.initializeFE(exampleName, physics, meshFile, matProp, overrideGPU, penal, nonDesignRegion);
topOpt.initializeOptimizer(numLayers, numNeuronsPerLyr, desiredVolumeFraction,localVolumeControl, fourierMap, densityProjection, maxLengthScaleControl, symXAxis, symYAxis);
topOpt.optimizeDesign(maxEpochs,minEpochs,useSavedNet);
modelWeights, modelBiases = topOpt.topNet.getWeights();
print("Time taken (secs): {:.2F}".format( time.perf_counter() - start))

