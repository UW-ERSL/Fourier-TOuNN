import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.collections

def to_torch(x):
    return torch.tensor(x).float()

def quadShapeFunction(xi, eta):
    N = 0.25*np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)]);
    gradN = 0.25*np.array([[eta-1, 1-eta, eta+1, -eta-1],[xi-1, -xi-1, xi+1, 1-xi]])
    return N, gradN

def jacobianQuad(xNodes, yNodes, xi, eta):
    J = np.zeros((2,2))
    N, gradN = quadShapeFunction(xi,eta);
    J[0,0] = np.dot(gradN[0,:], xNodes)
    J[0,1] = np.dot(gradN[1,:], xNodes)
    J[1,0] = np.dot(gradN[0,:], yNodes)
    J[1,1] = np.dot(gradN[1,:], yNodes)
    return J


class QuadMesh:
    def __init__(self, mesh, material, bc):
        self.numDOFPerNode = 2 # structural
        self.numDOFPerElem = 4*self.numDOFPerNode
        
        #%% Quadrature points
        self.xi_GQ = np.array([-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)]);
        self.eta_GQ = np.array([-1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]);
        self.wt_GQ = np.array([1.,1.,1.,1.])
        
        self.N = np.zeros((4,4))
        self.gradN = np.zeros((4,2,4))
        # Compute shape functions
        for i in range(len(self.xi_GQ)):
            self.N[i,:], self.gradN[i,:,:] = quadShapeFunction(self.xi_GQ[i], self.eta_GQ[i]);

        #%% Material
        self.material = material
        E = material['E'];
        nu = material['nu'];
        self.D = E/(1-nu**2)* np.array([[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2]]);

        #%% Read BC
        # Force
        with open(bc['force']) as f:
            self.f = np.array([float(line.rstrip()) for line in f]).reshape(-1,1);
        self.ndof = self.f.shape[0]
        # Fixed
        with open(bc['fixed']) as f:
            self.fixed = np.array([int(line.rstrip()) for line in f]).reshape(-1);
        self.free = np.setdiff1d(np.arange(self.ndof),self.fixed)
        
        #%% Connectivity info
        ctr = 0;
        f = open(mesh['elemNodes']);
        self.numElems = int(f.readline().rstrip());

        self.elemNodes = np.zeros((self.numElems, 4));
        self.edofMat = np.zeros((self.numElems,4*self.numDOFPerNode));
        for line in f:
            self.elemNodes[ctr,:] = line.rstrip().split('\t');
            self.edofMat[ctr,:] = np.array([[2*self.elemNodes[ctr,i],\
                                  2*self.elemNodes[ctr,i]+1] \
                                 for i in range(4) ]).reshape(-1);
            ctr += 1;
        self.edofMat = self.edofMat.astype(int)
        self.elemNodes = self.elemNodes.astype(int)
        
        self.iK = np.kron(self.edofMat,np.ones((4*self.numDOFPerNode,1))).flatten()
        self.jK = np.kron(self.edofMat,np.ones((1,4*self.numDOFPerNode))).flatten()
        #%% Read mesh info
        
        self.numNodes = int(self.ndof/self.numDOFPerNode)
        # Node XY
        self.nodeXY = np.zeros((self.numNodes,2));
        ctr = 0;
        f = open(mesh['nodeXY']);
        for line in f:
            self.nodeXY[ctr,:] = line.rstrip().split('\t')
            ctr += 1;

        # compute elemCenters, vertices and area
        self.elemCenters = np.zeros((self.numElems, 2));
        self.elemVertices = np.zeros((self.numElems,4,2));
        self.elemArea = np.zeros((self.numElems));
        for elem in range(self.numElems):
            nodes = ((self.edofMat[elem,0::2]+2)/2).astype(int)-1;
            for i in range(4):
                self.elemCenters[elem,0] += 0.25*self.nodeXY[nodes[i],0];
                self.elemCenters[elem,1] += 0.25*self.nodeXY[nodes[i],1];
            self.elemVertices[elem,:,0] = self.nodeXY[self.elemNodes[elem,:],0];
            self.elemVertices[elem,:,1] = self.nodeXY[self.elemNodes[elem,:],1];
            
            d1 = self.nodeXY[nodes[0],:] - self.nodeXY[nodes[2],:]
            d2 = self.nodeXY[nodes[1],:] - self.nodeXY[nodes[3],:]
            
            self.elemArea[elem] = 0.5*np.abs(d1[0]*d2[1] - d1[1]*d2[0])
        self.elemArea = to_torch(self.elemArea)
        self.netArea = torch.sum(self.elemArea)

        self.bb_xmin,self.bb_xmax,self.bb_ymin,self.bb_ymax =  np.min(self.nodeXY[:,0]),\
            np.max(self.nodeXY[:,0]),np.min(self.nodeXY[:,1]),np.max(self.nodeXY[:,1])

        #%% Elem stiffness matrices
        self.computeElemStiffMat()
    #-----------------------#
    #%% pts within mesh
    def generatePoints(self, res = 1):
        if(res == 1):
            x = np.array([0.])
        else:
            x = np.linspace(-1., 1., res)
        X,Y = np.meshgrid(x,x)
        isoPts = np.vstack((X.reshape(-1),Y.reshape(-1))).T

        N = np.zeros(((res)**2,4))
        gradN = np.zeros(((res)**2,2,4))
        for i in range(res**2):
            N[i,:], gradN[i,:,:] = quadShapeFunction(isoPts[i,0], isoPts[i,1])
        
        points = np.zeros((self.numElems*(res)**2,2))
        ctr = 0
        for elm in range(self.numElems):
            nodes = self.elemNodes[elm,:]
            xNodes, yNodes = self.nodeXY[nodes,0], self.nodeXY[nodes,1]
            for p in range(res**2):
                points[ctr,0] = xNodes[0]*N[p,0] + xNodes[1]*N[p,1] + \
                                xNodes[2]*N[p,2] + xNodes[3]*N[p,3]
                points[ctr,1] = yNodes[0]*N[p,0] + yNodes[1]*N[p,1] + \
                                yNodes[2]*N[p,2] + yNodes[3]*N[p,3]
                ctr += 1
        
        return points;
    #-----------------------#
    #%% Elem stiffness matrix
    def computeElemStiffMat(self): 
        self.KE = np.zeros((self.numElems, self.numDOFPerElem, self.numDOFPerElem))
        for elem in range (self.numElems):
            KElemtemp = np.zeros((self.numDOFPerElem, self.numDOFPerElem))
            nodes = self.elemNodes[elem,:]
            xNodes, yNodes = self.nodeXY[nodes,0], self.nodeXY[nodes,1]
            Z = np.zeros(int(self.numDOFPerElem/2))
            for g in range(len(self.wt_GQ)):
                gradN = self.gradN[g,:,:]
                J = jacobianQuad(xNodes, yNodes, self.xi_GQ[g], self.eta_GQ[g])
                dJ = np.abs(np.linalg.det(J))
                T = np.linalg.inv(J) @ gradN
                B = np.array([np.concatenate((T[0,:], Z)), np.concatenate((Z, T[1,:])), \
                            np.concatenate((T[1,:], T[0,:]))]);
                KElemtemp = KElemtemp + self.wt_GQ[g] * dJ * np.transpose(B) @ self.D @ B;
          
            order = [0, 4, 1, 5, 2, 6, 3, 7];
            KElemtemp = KElemtemp[:,order];
            KElemtemp = KElemtemp[order,:];
            self.KE[elem,:,:] = KElemtemp
    #-----------------------#
    def plotField(self, field, titleStr, res = 1):

        y = self.nodeXY[:,0]
        z = self.nodeXY[:,1]
    
        def quatplot(y,z, quatrangles, values, ax=None, **kwargs):
    
            if not ax: ax=plt.gca()
            yz = np.c_[y,z]
            verts= yz[quatrangles]
            pc = matplotlib.collections.PolyCollection(verts, **kwargs)
            pc.set_array(values)
            ax.add_collection(pc)
            ax.autoscale()
            return pc
    
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
    
        pc = quatplot(y,z, np.asarray(self.elemNodes), field, ax=ax, 
                 edgecolor="crimson", cmap="gray")
        # fig.colorbar(pc, ax=ax)        
        # ax.plot(y,z, marker="o", ls="", color="crimson")
        ax.set(title=titleStr, xlabel='X', ylabel='Y')
        plt.pause(0.001)
        plt.show()

        
      
          
  

# material = {'E':1.0, 'nu':0.3}; # Structural
# mesh = {'type':'quad',\
#         'nodeXY':'./Mesh/bridge/nodeXY.txt',\
#         'elemNodes':'./Mesh/bridge/elemNodes.txt'}

# bc = {'exampleName':'bridge', 'physics':'Structural', 'numDOFPerNode': 2,\
#   'force':'./Mesh/bridge/force.txt', 'fixed':'./Mesh/bridge/fixed.txt'};
    
# Q = QuadMesh(mesh, material, bc)
# Q.plotField(np.ones((Q.numElems)), 'test')