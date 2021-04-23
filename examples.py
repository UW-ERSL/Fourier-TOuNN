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
example = 1;
#  ~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~#
if(example == 1): # tip cantilever
    exampleName = 'TipCantilever'
    physics = 'Structural'
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed = dofs[0:2*(nely+1):1];
    force[2*(nelx+1)*(nely+1)-2*nely+1, 0 ] = -1;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 2): # mid cantilever
    exampleName = 'MidCantilever'
    physics = 'Structural';
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed = dofs[0:2*(nely+1):1];
    force[2*(nelx+1)*(nely+1)- (nely+1), 0 ] = -1;
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 3): #  MBBBeam
    exampleName = 'MBBBeam'
    physics = 'Structural'
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed= np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1);
    force[2*(nely+1)+1 ,0]=-1;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 4): #  Michell
    exampleName = 'Michell'
    physics = 'Structural'
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed=np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely] );
    force[nelx*(nely+1)+2 ,0]=-1;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':True, 'midPt':0.5*nelx};

elif(example == 5): #  DistributedMBB
    exampleName = 'Bridge'
    physics = 'Structural'
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed= np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1,2*(nelx+1)*(nely+1)-2*nely] );
    force[2*nely+1:2*(nelx+1)*(nely+1):8*(nely+1),0]=-1/(nelx+1);
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':True, 'midPt':0.5*nelx};

elif(example == 6): # Tensile bar
    exampleName = 'TensileBar'
    physics = 'Structural'
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof); 
    fixed =np.union1d(np.arange(0,2*(nely+1),2), 1); # fix X dof on left
    midDofX= 2*(nelx+1)*(nely+1)- (nely);
    force[midDofX, 0 ] = 1;
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 7): #  two fixed ends
    exampleName = 'twoFixedEnds'
    physics = 'Structural'
    ndof = 2*(nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof);
    fixed= np.union1d(np.arange(0,2*(nely+1),1), np.arange(2*(nely+1)*(nelx),2*(nely+1)*(nelx+1),1));
    force[nelx*(nely+1)+1 ,0]=-1;
    symXAxis = {'isOn':False, 'midPt':0.5*nely};
    symYAxis = {'isOn':True, 'midPt':0.5*nelx};

elif(example == 8): # Tapered bar
    exampleName = 'TaperedBar'
    physics = 'Thermal'
    ndof = (nelx+1)*(nely+1);
    force = np.zeros((ndof,1))
    dofs=np.arange(ndof); 
    fixed =np.union1d(np.arange(0,(nely+1),1), 1); # fix dofs on left
    midDofX= (nelx+1)*(nely+1)- int(0.5*nely); # right edge mid DOF
    force[midDofX-4:midDofX+4:1, 0 ] = 1;
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};
    
elif(example == 9):     
    exampleName = 'rightEdgeHeatLoad'
    physics = 'Thermal'
    ndof = (nelx+1)*(nely+1);
    force = np.zeros((ndof,1));
    force[np.arange(int(nelx*(nely+1)) , int((nelx+1)*(nely+1)) )] = 1;
    fixed =  int(nely/2 + 1 - nely/20)
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};

elif(example == 10): # BodyLoad
    exampleName = 'bodyHeatLoad'
    physics = 'Thermal'
    ndof = (nelx+1)*(nely+1);
    force = 0.*np.ones((ndof,1));
    force[0::1] = 2.5e-3
    fixed =  np.arange(int(nely/2 -5) , int(nely/2 + 5) );
    symXAxis = {'isOn':True, 'midPt':0.5*nely};
    symYAxis = {'isOn':False, 'midPt':0.5*nelx};
