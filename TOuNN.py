# FourierTOuNN: Length Scale Control in Topology Optimization, using Fourier Enhanced Neural Networks
# Authors : Aaditya Chandrasekhar, Krishnan Suresh
# Affliation : University of Wisconsin - Madison
# Corresponding Author : ksuresh@wisc.edu , achandrasek3@wisc.edu
# Submitted to Computer Aided Design, 2021
# For academic purposes only

#Versions
#Numpy 1.18.1
#Pytorch 1.5.0
#scipy 1.4.1
#cvxopt 1.2.0

#%% imports
import numpy as np
import torch
import torch.optim as optim
from os import path
from FE import FE
from extrusion import applyExtrusion
import matplotlib.pyplot as plt
from network import TopNet
from torch.autograd import grad
from gridMesher import GridMesh

def to_np(x):
    return x.detach().cpu().numpy()
#%% main TO functionalities
class TopologyOptimizer:
    #-----------------------------#
    def __init__(self, mesh, matProp, bc, nnSettings, fourierMap, \
                  desiredVolumeFraction, densityProjection, keepElems, extrude, overrideGPU = True):

        self.exampleName = bc['exampleName'];
        self.device = self.setDevice(overrideGPU);
        self.FE = FE(mesh, matProp, bc)
        xy = self.FE.mesh.generatePoints()
        self.xy = torch.tensor(xy, requires_grad = True).\
                                        float().view(-1,2).to(self.device);
        self.keepElems = keepElems
        self.desiredVolumeFraction = desiredVolumeFraction;
        self.density = self.desiredVolumeFraction*np.ones((self.FE.mesh.numElems));
        self.symXAxis = bc['symXAxis'];
        self.symYAxis = bc['symYAxis'];
        self.fourierMap = fourierMap;
        self.extrude = extrude;
        self.densityProjection = densityProjection;
        if(self.fourierMap['isOn']):
            coordnMap = np.zeros((2, self.fourierMap['numTerms']));
            for i in range(coordnMap.shape[0]):
                for j in range(coordnMap.shape[1]):
                    coordnMap[i,j] = np.random.choice([-1.,1.])*np.random.uniform(1./(2*fourierMap['maxRadius']), 1./(2*fourierMap['minRadius']));  #

            self.coordnMap = torch.tensor(coordnMap).float().to(self.device)#
            inputDim = 2*self.coordnMap.shape[1];
        else:
            self.coordnMap = torch.eye(2);
            inputDim = 2;
        self.topNet = TopNet(nnSettings, inputDim).to(self.device);
        self.objective = 0.;
    #-----------------------------#
    def setDevice(self, overrideGPU):
        if(torch.cuda.is_available() and (overrideGPU == False) ):
            device = torch.device("cuda:0");
            print("GPU enabled")
        else:
            device = torch.device("cpu")
            print("Running on CPU")
        return device;

    #-----------------------------#
    def applySymmetry(self, x):
        if(self.symYAxis['isOn']):
            xv =( self.symYAxis['midPt'] + torch.abs( x[:,0] - self.symYAxis['midPt']));
        else:
            xv = x[:,0];
        if(self.symXAxis['isOn']):
            yv = (self.symXAxis['midPt'] + torch.abs( x[:,1] - self.symXAxis['midPt'])) ;
        else:
            yv = x[:,1];
        x = torch.transpose(torch.stack((xv,yv)),0,1);
        return x;
    #-----------------------------#
    def applyFourierMapping(self, x):
        if(self.fourierMap['isOn']):
            c = torch.cos(2*np.pi*torch.matmul(x,self.coordnMap));
            s = torch.sin(2*np.pi*torch.matmul(x,self.coordnMap));
            xv = torch.cat((c,s), axis = 1);
            return xv;
        return x;
    #-----------------------------#
    def projectDensity(self, x):
        if(self.densityProjection['isOn']):
            b = self.densityProjection['sharpness']
            nmr = np.tanh(0.5*b) + torch.tanh(b*(x-0.5));
            x = 0.5*nmr/np.tanh(0.5*b);
        return x;
    #-----------------------------#
    def optimizeDesign(self,maxEpochs, minEpochs):
        self.convergenceHistory = {'compliance':[], 'vol':[], 'grayElems':[]};
        learningRate = 0.01;
        alphaMax = 100
        alphaIncrement= 0.2;
        alpha = alphaIncrement; # start
        nrmThreshold = 0.01; # for gradient clipping
        self.optimizer = optim.Adam(self.topNet.parameters(), amsgrad=True,lr=learningRate);
        print("Iter \t Obj \t vol \t relGray \n")
        for epoch in range(maxEpochs):
            self.optimizer.zero_grad();
            batch_x = self.applySymmetry(self.xy);
            x = self.applyFourierMapping(batch_x);
            nn_rho = torch.flatten(self.topNet(x)).to(self.device);
            nn_rho = self.projectDensity(nn_rho);
            rhoElem = applyExtrusion(nn_rho, self.extrude)

            rhoElem[self.keepElems['idx']] = self.keepElems['density']
            self.density = to_np(rhoElem);

            u, Jelem = self.FE.solve(self.density); # Call FE 88 line code [Niels Aage 2013]

            if(epoch == 0):
                self.obj0 = ((0.01+self.density)**(2*self.FE.mesh.material['penal'])*Jelem).sum()
            # For sensitivity analysis, exponentiate by 2p here and divide by p in the loss func hence getting -ve sign

            Jelem = np.array((self.density**(2*self.FE.mesh.material['penal']))*Jelem).reshape(-1);
            Jelem = torch.tensor(Jelem).view(-1).float().to(self.device) ;
            objective = torch.sum(torch.div(Jelem,rhoElem**self.FE.mesh.material['penal']))/self.obj0; # compliance

            volConstraint =((torch.sum(self.FE.mesh.elemArea*rhoElem)/\
                             (self.FE.mesh.netArea*self.desiredVolumeFraction)) - 1.0); # global vol constraint
            currentVolumeFraction = np.average(self.density);
            self.objective = objective;
            loss =   self.objective+ alpha*torch.pow(volConstraint,2) #+ 5.*alpha*torch.pow(keepElemLoss,2);

            alpha = min(alphaMax, alpha + alphaIncrement);
            loss.backward(retain_graph=True);
            torch.nn.utils.clip_grad_norm_(self.topNet.parameters(),nrmThreshold)
            self.optimizer.step();

            greyElements= sum(1 for rho in self.density if ((rho > 0.2) & (rho < 0.8)));
            relGreyElements = self.desiredVolumeFraction*greyElements/len(self.density);
            self.convergenceHistory['compliance'].append(self.objective.item());
            self.convergenceHistory['vol'].append(currentVolumeFraction);
            self.convergenceHistory['grayElems'].append(relGreyElements);
            self.FE.mesh.material['penal'] = min(8.0,self.FE.mesh.material['penal'] + 0.02); # continuation scheme

            if(epoch % 30 == 0):
                titleStr = "{:d} \t {:.2F} \t {:.2F} \t {:.4F}".\
                    format(epoch, self.objective.item()*self.obj0,\
                           currentVolumeFraction, relGreyElements)
                #self.FE.mesh.plotField(-self.density, titleStr)
                print(titleStr);
            if ((epoch > minEpochs ) & (relGreyElements < 0.0025)):
                break;
        self.FE.mesh.plotField(-self.density, titleStr)
        print("{:3d} J: {:.2F}; Vf: {:.3F}; loss: {:.3F}; relGreyElems: {:.3F} "\
             .format(epoch, self.objective.item()*self.obj0 ,currentVolumeFraction,loss.item(),relGreyElements));

        print("Final J : {:.3f}".format(self.objective.item()*self.obj0));
        self.plotConvergence(self.convergenceHistory);

        return self.convergenceHistory;
    #-----------------------------#
    def postProcessHighRes(self, mesh = None, keepElems = None):
        if(mesh == None):
            mesh = self.FE.mesh
        if(keepElems == None):
            keepElems = self.keepElems
        xy = torch.tensor(mesh.elemCenters, requires_grad = True).view(-1,2).float()
        batch_x = self.applySymmetry(xy);
        x = self.applyFourierMapping(batch_x);
        rho = torch.flatten(self.projectDensity(self.topNet(x)));
        rho[keepElems['idx']] = keepElems['density']
        rho_np = rho.cpu().detach().numpy();
        titleStr = "Obj {:.2F} , vol {:.2F}".format(\
                    self.objective.item()*self.obj0, np.mean(rho_np));
            
        mesh.plotField(-rho_np, titleStr)
        
    #-----------------------------#
    def plotConvergence(self, convg):
        plt.figure();
        for key in convg:
            y = np.array(convg[key]);
            plt.semilogy(y, label = str(key));
            plt.xlabel('Iterations');
            plt.ylabel(str(key));
            plt.grid('True')
            plt.figure();
