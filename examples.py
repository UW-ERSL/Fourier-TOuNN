import numpy as np
import time
from TOuNN import TopologyOptimizer 
from extrusion import computeExtrusionMap
from gridMesher import GridMesh
import matplotlib.pyplot as plt
#  ~~~~~~~~~~~~ Setup ~~~~~~~~~~~~~#
example = 5; # see below for description
#  ~~~~~~~~~~~~Material~~~~~~~~~~~~~#

matProp = {'E':1.0, 'nu':0.3, 'penal':1.}; # Structural, starting penal vals

#  ~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~#
if(example == 1): # tip cantilever
    exampleName = 'TipCantilever'
    nelx, nely = 60, 30
    elemSize = np.array([1.0,1.0]);
    mesh = {'type':'grid','nelx':nelx, 'nely':nely, 'elemSize':elemSize}
    physics = 'Structural'
    numDOFPerNode = 2
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed = dofs[0:2*(nely+1):1];
    force[2*(nelx+1)*(nely+1)-2*nely+1, 0 ] = -1;
    desiredVolumeFraction = 0.75;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};
    keepElems = {'idx':[], 'density':1.}
    extrude = {'X':{'isOn':False},'Y':{'isOn':False} }
    LMin, LMax = 6,30;
    numTerms = 150;
    bc = {'exampleName':exampleName, 'physics':physics, 'numDOFPerNode': numDOFPerNode,\
      'force':force, 'fixed':fixed, 'symXAxis':symXAxis, 'symYAxis':symYAxis };
elif(example == 2): # mid cantilever
    exampleName = 'MidCantilever'
    nelx, nely = 60, 30
    elemSize = np.array([1.0,1.0]);
    mesh = {'type':'grid','nelx':nelx, 'nely':nely, 'elemSize':elemSize}
    physics = 'Structural'
    numDOFPerNode = 2
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed = dofs[0:2*(nely+1):1];
    force[2*(nelx+1)*(nely+1)- (nely+1), 0 ] = -1;
    desiredVolumeFraction = 0.6;
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};
    keepElems = {'idx':[], 'density':1.}
    extrude = {'X':{'isOn':False},'Y':{'isOn':False} }
    LMin, LMax = 6,30
    numTerms = 150
    bc = {'exampleName':exampleName, 'physics':physics, 'numDOFPerNode': numDOFPerNode,\
      'force':force, 'fixed':fixed, 'symXAxis':symXAxis, 'symYAxis':symYAxis };
elif(example == 3): #  MBBBeam
    exampleName = 'MBBBeam'
    nelx, nely = 60, 30
    elemSize = np.array([1.0,1.0]);
    mesh = {'type':'grid','nelx':nelx, 'nely':nely, 'elemSize':elemSize}
    physics = 'Structural'
    numDOFPerNode = 2
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed= np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1);
    force[2*(nely+1)+1 ,0]=-1;
    desiredVolumeFraction = 0.45;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};
    extrude = {'X':{'isOn':False},'Y':{'isOn':False} }
    keepElems = {'idx':[], 'density':1.}
    LMin, LMax = 6, 30
    numTerms = 150
    bc = {'exampleName':exampleName, 'physics':physics, 'numDOFPerNode': numDOFPerNode,\
      'force':force, 'fixed':fixed, 'symXAxis':symXAxis, 'symYAxis':symYAxis };
elif(example == 4): #  Michell
    desiredVolumeFraction = 0.3; # between 0.1 and 0.9
    exampleName = 'Michell'
    nelx, nely = 60, 30
    elemSize = np.array([1.0,1.0]);
    mesh = {'type':'grid','nelx':nelx, 'nely':nely, 'elemSize':elemSize}
    physics = 'Structural'
    numDOFPerNode = 2
    meshType = 'rectGeom';
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed=np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1,2*(nelx+1)*(nely+1)-2*nely] ); # ,2*(nelx+1)*(nely+1)-2*nely+1,
    force[nelx*(nely+1)+1 ,0]=-1;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':True, 'midPt':0.5*nelx};
    extrude = {'X':{'isOn':False},'Y':{'isOn':False} }
    keepElems = {'idx':[], 'density':1.}
    LMin, LMax = 6, 30
    numTerms = 150
    bc = {'exampleName':exampleName, 'physics':physics, 'numDOFPerNode': numDOFPerNode,\
      'force':force, 'fixed':fixed, 'symXAxis':symXAxis, 'symYAxis':symYAxis };
elif(example == 5): #  Bridge
    exampleName = 'Bridge'
    physics = 'Structural'
    nelx, nely = 60, 30
    elemSize = np.array([1.0,1.0]);
    mesh = {'type':'grid','nelx':nelx, 'nely':nely, 'elemSize':elemSize}
    numDOFPerNode = 2
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed= np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1,2*(nelx+1)*(nely+1)-2*nely] );
    force[2*nely+1:2*(nelx+1)*(nely+1):2*(nely+1),0]=-1/(nelx+1);
    desiredVolumeFraction = 0.45;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':True, 'midPt':0.5*nelx};
    keepElems = {'idx':np.arange(nely-1, nelx*nely, nely), 'density':1.}
    extrude = {'X':{'isOn':False},'Y':{'isOn':False} }
    LMin, LMax = 4, 30;
    numTerms = 150
    bc = {'exampleName':exampleName, 'physics':physics, 'numDOFPerNode': numDOFPerNode,\
      'force':force, 'fixed':fixed, 'symXAxis':symXAxis, 'symYAxis':symYAxis };
elif(example == 6): # Tensile bar
    exampleName = 'TensileBar'
    nelx, nely = 60, 30
    elemSize = np.array([1.0,1.0]);
    mesh = {'type':'grid','nelx':nelx, 'nely':nely, 'elemSize':elemSize}
    physics = 'Structural'
    numDOFPerNode = 2
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof); 
    fixed =np.union1d(np.arange(0,2*(nely+1),2), 1); # fix X dof on left
    force[2*(nelx+1)*(nely+1)-2*nely:2*(nelx+1)*(nely+1):2, 0 ] = 0.1;
    desiredVolumeFraction = 0.4;
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};
    keepElems = {'idx':np.arange(nelx*(nely-1), nelx*nely), 'density':1.}
    extrude = {'X':{'isOn':True},'Y':{'isOn':False} }
    extrude = computeExtrusionMap(mesh, extrude)
    LMin, LMax = 2,4; # (2,4) (8, 10) (16, 20)
    numTerms = 150
    bc = {'exampleName':exampleName, 'physics':physics, 'numDOFPerNode': numDOFPerNode,\
      'force':force, 'fixed':fixed, 'symXAxis':symXAxis, 'symYAxis':symYAxis };
elif(example == 7):
    exampleName = 'LBracket'
    nelx, nely = 100, 100
    elemSize = np.array([1.0,1.0]);
    mesh = {'type':'grid','nelx':nelx, 'nely':nely, 'elemSize':elemSize}
    physics = 'Structural'
    numDOFPerNode = 2
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    forceDof = 2*(nelx+1)*(nely+1)- int(0.6*nely*2)-1
    force[forceDof, 0 ] = -1;
    dofs=np.arange(ndof); 
    fixed = np.union1d(np.arange(2*nely+1,(nelx+1)*(nely+1),2*(nely+1)),\
                       np.arange(2*nely,(nelx+1)*(nely+1),2*(nely+1)))
    desiredVolumeFraction = 0.4*0.64;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};
    elems = np.array([], dtype=np.int32)
    for col in range(int(0.4*nelx),nelx):
        start = col*nely + int(0.4*nely)
        stop = start + int(0.6*nely)
        elems = np.append(elems, np.arange(start, stop))
        
    keepElems = {'idx':elems, 'density':0.01}
    extrude = {'X':{'isOn':False},'Y':{'isOn':False} }
    extrude = computeExtrusionMap(mesh, extrude)
    LMin, LMax = 4, 30;
    numTerms = 250
    bc = {'exampleName':exampleName, 'physics':physics, 'numDOFPerNode': numDOFPerNode,\
      'force':force, 'fixed':fixed, 'symXAxis':symXAxis, 'symYAxis':symYAxis };

nnSettings = {'numLayers':1, 'numNeuronsPerLyr':20 }
fourierMap = {'isOn':True, 'minRadius':LMin, \
              'maxRadius':LMax, 'numTerms':numTerms};
densityProjection = {'isOn':True, 'sharpness': 6};
minEpochs = 150; # minimum number of iterations
maxEpochs = 500; # Max number of iterations

plt.close('all');
overrideGPU = True
start = time.perf_counter()
topOpt = TopologyOptimizer(mesh, matProp, bc, nnSettings, fourierMap, \
          desiredVolumeFraction, densityProjection, keepElems, extrude, overrideGPU);
topOpt.optimizeDesign(maxEpochs,minEpochs);
print("Time taken (secs): {:.2F}".format( time.perf_counter() - start))

#%% Post processing
res = 4
nelxHres, nelyHres = nelx*res, nely*res
elemSize = np.array([1.0/res,1.0/res]);
meshSpec = {'type':'grid','nelx':nelxHres, 'nely':nelyHres, 'elemSize':elemSize}
mesh = GridMesh(meshSpec)
if(example == 5):
    idx = []
    for elm in range(mesh.numElems):
        if((mesh.elemCenters[elm,0] > 0) and (mesh.elemCenters[elm,0] < nelx) and \
           (mesh.elemCenters[elm,1] > nely-1) and (mesh.elemCenters[elm,1] < nely)):
            idx.append(elm)
    den = 1.
    keepElems = {'idx':np.array(idx), 'density':1.}
if(example == 6):
    keepElems = {'idx':np.arange(nelxHres*(nelyHres-1), nelxHres*nelyHres), 'density':1.}
if(example == 7):
    idx = []
    for elm in range(mesh.numElems):
        if((mesh.elemCenters[elm,0] > 0.4*nelx) and (mesh.elemCenters[elm,0] < nelx) and \
           (mesh.elemCenters[elm,1] > 0.4*nely) and (mesh.elemCenters[elm,1] < nely)):
            idx.append(elm)
    den = 0.01
    keepElems = {'idx':np.array(idx), 'density':0.}
topOpt.postProcessHighRes(mesh, keepElems)
