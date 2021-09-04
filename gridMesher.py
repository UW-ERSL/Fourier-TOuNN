import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors

class GridMesh:
    def __init__(self, mesh, material = None, bc = None):
        self.mesh = mesh
        self.initMesh()
        if(bc != None):
            self.bc = bc
            self.initBC()
        if(material != None):
            self.material = material
            self.initK()
    #-----------------------#
    def initMesh(self):
        self.nelx = self.mesh['nelx']
        self.nely = self.mesh['nely']
        self.elemSize = self.mesh['elemSize']
        self.numElems = self.nelx*self.nely
        self.numNodes = (self.nelx+1)*(self.nely+1)
        self.elemNodes = np.zeros((self.numElems, 4));
        self.elemArea = self.elemSize[0]*self.elemSize[1]*torch.ones((self.numElems))
        self.netArea = torch.sum(self.elemArea)

        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely+elx*self.nely;
                n1=(self.nely+1)*elx+ely
                n2=(self.nely+1)*(elx+1)+ely
                self.elemNodes[el,:] = np.array([n1+1, n2+1, n2, n1]);
        self.elemNodes = self.elemNodes.astype(int)

        self.nodeXY = np.zeros((self.numNodes,2))
        ctr = 0;
        for i in range(self.nelx+1):
            for j in range(self.nely+1):
                self.nodeXY[ctr,0] = self.elemSize[0]*i;
                self.nodeXY[ctr,1] = self.elemSize[1]*j;
                ctr += 1;

        self.elemCenters = self.generatePoints()
        self.bb_xmin,self.bb_xmax,self.bb_ymin,self.bb_ymax =\
            0.,self.nelx*self.elemSize[0],0., self.nely*self.elemSize[1];

    #-----------------------#
    def initBC(self):
        self.ndof = self.bc['numDOFPerNode']*(self.nelx+1)*(self.nely+1);
        self.fixed = self.bc['fixed'];
        self.free = np.setdiff1d(np.arange(self.ndof),self.fixed);
        self.f = self.bc['force']
        self.numDOFPerElem = 4*self.bc['numDOFPerNode']
        self.edofMat=np.zeros((self.nelx*self.nely,self.numDOFPerElem),dtype=int);

        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely+elx*self.nely
                n1=(self.nely+1)*elx+ely
                n2=(self.nely+1)*(elx+1)+ely
                self.edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
        self.edofMat = self.edofMat.astype(int)
        
        self.iK = np.kron(self.edofMat,np.ones((self.numDOFPerElem ,1))).flatten()
        self.jK = np.kron(self.edofMat,np.ones((1,self.numDOFPerElem))).flatten()
        bK = tuple(np.zeros((len(self.iK))).astype(int)) #batch values
        self.nodeIdx = [bK,self.iK,self.jK]
        

    #-----------------------#
    def initK(self):
        def getDMatrix(materialProperty):

            E= 1
            nu= materialProperty['nu'];
            k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
            KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
            return (KE); #

        self.KE = np.tile(getDMatrix(self.material)[np.newaxis,:,:], (self.numElems,1,1))
    #-----------------------#
    def generatePoints(self,  resolution = 1): # generate points in elements
        ctr = 0;
        xy = np.zeros((resolution*self.nelx*resolution*self.nely,2));

        for i in range(resolution*self.nelx):
            for j in range(resolution*self.nely):
                xy[ctr,0] = self.elemSize[0]*(i + 0.5)/resolution;
                xy[ctr,1] = self.elemSize[1]*(j + 0.5)/resolution;
                ctr += 1;

        return xy;
    #-----------------------#
    def plotField(self, field, titleStr, res = 1):
        fig, ax = plt.subplots();
        plt.subplot(1,1,1);
        plt.imshow(field.reshape((res*self.nelx, res*self.nely)).T, cmap='gray',\
                    interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0), origin = 'lower')
        plt.axis('Equal')
        plt.title(titleStr)
        plt.grid(False)
        fig.canvas.draw();
        plt.pause(0.01)
        
