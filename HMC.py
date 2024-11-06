import torch as tr
import numpy as np
from tqdm import tqdm
import sys
import os


def autograd(func,x):
    x_tensor = x.clone().detach()
    x_tensor.requires_grad_()
    #detect anomaly
    #x_tensor.set_anomaly_enabled(True)
    y = func(x_tensor)
    y.backward()
    return x_tensor.grad

class HMC_sampler():
    def __init__(self,logfunc,diagonal,device,grad=None):
        self.logfunc = logfunc
        self.device=device
        self.diagonal = diagonal.to(self.device)
        self.grad = grad
    def gibbs(self):
        self.p0 = tr.distributions.MultivariateNormal(tr.zeros_like(self.diagonal), covariance_matrix=tr.diag(self.diagonal)).sample().to(self.device)
        return self.p0

    def HMC_1(self,q0,p0,eps,L):
        self.eps=eps
        self.L=L
        self.q0=q0.to(self.device)
        
        q = q0
        p = p0
        #CURRENT HAMILTONIAN
        H0=self.hamiltonian(q0,p0)
        if self.grad is None:
            #grad_U = -autograd(self.logfunc,q)
            p = p - self.eps/2.0 * autograd(self.logfunc,q)
            for i in range(self.L):
                if tr.isnan(q).any() and tr.isnan(p).any():
                    #print('rejected in leapfrog')
                    return q0,p0,0
                q = q + self.eps * p
                if i!=self.L-1:
                    if tr.isinf(self.logfunc(q)).item():
                        #print('rejected in leapfrog 2')
                        return q0,p0,0
                    else:
                        p = p - self.eps *autograd(self.logfunc,q)
            p = p - self.eps/2.0 * autograd(self.logfunc,q)
            #reversibility
            #p = -p
        else:
            p = p - self.eps/2.0 * self.grad(q)
            for i in range(self.L):
                q = q + self.eps * p
                if i!=self.L-1:
                    if  tr.isinf(self.logfunc(q)).item():
                        #print('rejected in leapfrog 2')
                        return q0,p0,0
                    else:
                        p = p - self.eps * self.grad(q)
            p = p - self.eps/2.0 * self.grad(q)
            #reversibility
            #p = -p

        #PROPOSED HAMILTONIAN
        H1=self.hamiltonian(q,p)
        #delta H= H0-H1
        ΔH=H0-H1


        prob=tr.min(tr.tensor([1.0,tr.exp(ΔH)]))
        if tr.isnan(q).any() and tr.isnan(p).any():
                #print('rejected after leapfrog')
                return q0,p0,ΔH

        if np.random.uniform(0,1.0) < prob.item():
            self.flag=True
            return q,p,ΔH
        else:
            self.flag=False
            #print('rejected')
            return q0,p0,ΔH
        
    def hamiltonian(self,q,p):
        return self.logfunc(q) + tr.sum(p**2)/2.0
    
    def sample(self,q0,Nsamp,eps,L):
        self.traceq = tr.zeros(Nsamp,q0.shape[0]).to(self.device)
        self.tracep = tr.zeros(Nsamp,q0.shape[0]).to(self.device)
        self.traceH = tr.zeros(Nsamp,1)
        self.traceq[0] = self.q0
        self.p0 = self.gibbs()
        self.tracep[0] = self.p0
        self.eps=eps
        self.L=L
        for i in tqdm(range(1,Nsamp),file=sys.stdout,miniters=Nsamp//100, maxinterval=float("inf")):
            self.p0 = self.gibbs()
            q,p,ΔH = self.HMC_1(self.traceq[i-1],self.p0,self.eps,self.L)
                
            self.traceq[i] = q
            self.tracep[i] = p
            self.traceH[i] = ΔH
        return self.traceq,self.tracep,self.traceH
    

    def epsilonsqtest(self,ini,fin,NL,q0):
        #create new variables
        #eps0=self.eps
        self.q0= q0
        p0=self.gibbs()
        LT=(tr.linspace(ini,fin,NL)).type(tr.int32)
        epsT=(1/LT)
        #H0=self.hamiltonian(q0,p0)
        ΔH=tr.zeros(epsT.shape[0])
        i=0
        print(epsT,LT)
        while i<(epsT.shape[0]):
            
            self.eps=epsT[i]
            self.L=LT[i]
            
            #H0=self.hamiltonian(q0,p0)
            q,p,ΔH1 = self.HMC_1(q0,p0,epsT[i],LT[i])
            if self.flag:
                ΔH[i]=ΔH1
                print(epsT[i],ΔH[i])
                i=i+1
            else:
                print('rejected')
                continue
        #self.eps=eps0
        return epsT,ΔH
    
