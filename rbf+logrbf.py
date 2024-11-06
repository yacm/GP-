import torch as tr

if tr.backends.mps.is_available():
    device = tr.device("mps")
    x = tr.ones(1, device=device)
    #print (x)
elif tr.cuda.is_available():
    device = tr.device("cuda")
    x = tr.ones(1, device=device)
    #print (x)
else:
    device = tr.device("cpu")

from GP import *
import torch as tr
import numpy as np
import matplotlib.pyplot as plt
import argparse 

#parser

parser = argparse.ArgumentParser(description='HMC sampler for Gaussian Process')
parser.add_argument('--i', type=int, help='data set to analize 0-11 (mock-data=12)')
parser.add_argument('--Nsamples', type=int, help='number of samples')
parser.add_argument('--burn', type=int, default=0, help='burn in')
parser.add_argument('--L', type=int, default=100, help='number of leapfrog steps')
parser.add_argument('--eps', type=float, default=1.0/1000, help='step size')
args = parser.parse_args()

print(args)
i=args.i
Nsamples=args.Nsamples
burn=args.burn
L=args.L
eps=args.eps


import scipy.integrate as integrate
from torch.special import gammaln
#from orthogonal_poly import legendre_01

from torch.autograd.functional import hessian

import scipy.integrate as integrate

import h5py as h5

# import all packages and set plots to be embedded inline
import numpy as np 
from scipy.optimize import minimize 
from scipy import special 
from scipy.optimize import Bounds 
from scipy.linalg import cho_solve 
from pyDOE import lhs 
import time
from sklearn.preprocessing import MinMaxScaler 
from sklearn.pipeline import Pipeline 
import torch as tr
import scipy.special

def get_dist_matelem(z, p, t_min):
    f = 0
    if p <= 3:
        f = h5.File('pdf-data/Nf2+1/ratio.summationLinearFits.cl21_32_64_b6p3_m0p2350_m0p2050.unphased.hdf5','r')
    else:
        f = h5.File('pdf-data/Nf2+1/ratio.summationLinearFits.cl21_32_64_b6p3_m0p2350_m0p2050.phased-d001_2.00.hdf5','r')
    M_z_p = np.array(f['MatElem/bins/Re/mom_0_0_+'+str(p)+'/disp_z+'+str(z)+'/insertion_gt/tsep_'+str(t_min)+'-14'])
    M_0_0 = np.array(f['MatElem/bins/Re/mom_0_0_0/disp_0/insertion_gt/tsep_'+str(t_min)+'-14'])
    M_z_0 = np.array(f['MatElem/bins/Re/mom_0_0_0/disp_z+'+str(z)+'/insertion_gt/tsep_'+str(t_min)+'-14'])
    M_0_p = np.array(f['MatElem/bins/Re/mom_0_0_+'+str(p)+'/disp_0/insertion_gt/tsep_'+str(t_min)+'-14'])
    
    f.close()
    return M_z_p * M_0_0 / M_0_p / M_z_0

def get_final_res(z, p):
    m_4, _ = get_dist_matelem(z, p, 4)
    m_6, s_6 = get_dist_matelem(z, p, 6)
    m_8, s_8 = get_dist_matelem(z, p, 8)
    return m_6, np.sqrt(s_6**2)#+(m_4-m_6)**2)

Np = 6
Nz = 12
Nj = 349
rMj = np.empty([Nj,Np,Nz])
nu = np.empty([Np,Nz])
for p in range(1,Np+1):
    for z in range (1,Nz+1):
        nu[p-1,z-1] = 2.0*np.pi/32.0 *p *z
        #print(p,z,nu[p-1,z-1])
        m_4 = get_dist_matelem(z,p,4)
        m_6 = get_dist_matelem(z,p,6)
        m_8 = get_dist_matelem(z,p,8)
        #expo fit
        m = (m_4*m_8 - m_6**2)/(m_4 + m_8 - 2 * m_6)
        # this fails for certain cases where the denomenator goes too close to zero
        # use the m_6 as default
        rMj[:,p-1,z-1] = m_6
        #Nj=m.shape[0]
        #print(z,p,np.mean(m_4),np.mean(m_6),np.mean(m_8), np.mean(m),np.std(m)*np.sqrt(Nj-1))
rM = np.mean(rMj,axis=0)
rMe = np.std(rMj,axis=0)*np.sqrt(Nj) 
#plot the data
#for i in range(0,6):
#    plt.errorbar(nu[i],rM[i],yerr=rMe[i],fmt='.',alpha=0.5,label='p='+str(i+1))
#plt.legend()
#plt.show()
##integrator
class FE_Integrator:
    def __init__(self,x):
        self.N = x.shape[0]
        xx = np.append(x,2.0*x[self.N-1] - x[self.N-2])
        self.x = np.append(0,xx)
        self.eI = 0

        self.Norm = np.empty(self.N)
        for i in range(self.N):
            self.Norm[i] = self.ComputeI(i, lambda x : 1)
            
    def pulse(self,x,x1,x2):
        return np.heaviside(x-x1,0.5)* np.heaviside(x2-x,0.5)
    
    def f(self,x,i):
 ##       if(i==0):
 ##           R=(x- self.x[2])/(self.x[1] -self.x[2])*np.heaviside(x-self.x[0],1.0)* np.heaviside(self.x[2]-x,0.5)

            #R= self.pulse(x,self.x[0],self.x[1])
            #R= (x- self.x[0])/(self.x[1] -self.x[0])*self.pulse(x,self.x[0],self.x[1])
            #R+=(x- self.x[2])/(self.x[1] -self.x[2])*self.pulse(x,self.x[1],self.x[2])
            #R+=(x- self.x[1])/(self.x[0] -self.x[1])*self.pulse(x,self.x[0],self.x[1]) 
##            return R
        ii=i+1
        R = (x- self.x[ii-1])/(self.x[ii] -self.x[ii-1])*self.pulse(x,self.x[ii-1],self.x[ii  ])
        R+= (x- self.x[ii+1])/(self.x[ii] -self.x[ii+1])*self.pulse(x,self.x[ii  ],self.x[ii+1])

       # if(i==0):
       #     R *=2
        return R
    
    def set_up_integration(self,Kernel = lambda x: 1):
        res = np.empty(self.N)
        for i in range(self.N):
            res[i] = self.ComputeI(i,Kernel)
        return res
   
    # assume symmetrix function F(x,y) = F(y,x)
    # for efficiency
    def set_up_dbl_integration(self,Kernel = lambda x,y: 1):
        res = np.empty([self.N,self.N])
        for i in range(self.N):
            for j in range(i,self.N):
                res[i,j] = self.ComputeIJ(i,j,Kernel)
                res[j,i]  = res[i,j]
        #res[0,:] *=2
        #res[:,0] *=2
        return res
        
    def ComputeI(self,i,Kernel):
        I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,i), self.x[i], self.x[i+2])
        self.eI += eI
        return I
    
    def ComputeIJ(self,i,j,Kernel):
        I,eI = integrate.dblquad(lambda x,y: self.f(x,i)*Kernel(x,y)*self.f(y,j), self.x[j], self.x[j+2],self.x[i], self.x[i+2])
        self.eI += eI
        return I
    
    
# quadratic finite elements are more complicated...
# ... but now it works!
# also I should try the qubic ones too
class FE2_Integrator:
    def __init__(self,x):
        self.N = x.shape[0]
        xx = np.append(x,[2.0*x[self.N-1] - x[self.N-2], 3.0*x[self.N-1]-2*x[self.N-2],0] )
        #self.x = np.append([-x[0],0],xx)
        self.x = np.append(0,xx)
        self.eI = 0

        self.Norm = np.empty(self.N)
        for i in range(self.N):
            self.Norm[i] = self.ComputeI(i, lambda x : 1)
            
    def pulse(self,x,x1,x2):
        return np.heaviside(x-x1,0.5)* np.heaviside(x2-x,0.5)
    
    def f(self,x,i):
        R=0.0
        if(i==0):
            #R=self.pulse(x,self.x[0],self.x[1])
            #R=self.pulse(x,self.x[1],self.x[2])
        #    R+=(x- self.x[2])/(self.x[1] -self.x[2])*self.pulse(x,self.x[1],self.x[2])

            R+=(x- self.x[2])*(x- self.x[3])/((self.x[1] -self.x[3])*(self.x[1] -self.x[2]))**np.heaviside(x-self.x[0],1.0)* np.heaviside(self.x[3]-x,0.5)
            #self.pulse(x,self.x[0],self.x[3])
            return R
        ii =i+1
        if(ii%2==0):
            R  += (x- self.x[ii-1])*(x- self.x[ii+1])/((self.x[ii] -self.x[ii+1])*(self.x[ii] -self.x[ii-1]))*self.pulse(x,self.x[ii-1],self.x[ii+1])
            return R
        else:
            R += (x- self.x[ii-2])*(x- self.x[ii-1])/((self.x[ii] -self.x[ii-2])*(self.x[ii] -self.x[ii-1]))*self.pulse(x,self.x[ii-2],self.x[ii  ])
            R += (x- self.x[ii+1])*(x- self.x[ii+2])/((self.x[ii] -self.x[ii+2])*(self.x[ii] -self.x[ii+1]))*self.pulse(x,self.x[ii  ],self.x[ii+2])
            return R
    
        return R
    
    def set_up_integration(self,Kernel = lambda x: 1):
        res = np.empty(self.N)
        for i in range(self.N):
            res[i] = self.ComputeI(i,Kernel)
        return res
        
    # assume symmetrix function F(x,y) = F(y,x)
    # for efficiency 
    def set_up_dbl_integration(self,Kernel = lambda x,y: 1):
        res = np.empty([self.N,self.N])
        for i in range(self.N):
            for j in range(i,self.N):
                res[i,j] = self.ComputeIJ(i,j,Kernel)
                res[j,i]  = res[i,j]
        return res
    
    def ComputeI(self,i,Kernel):
        #if(i==0):
        #    I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,0), self.x[0], self.x[3])
        #    self.eI += eI
        #    return I
        ii=i+1
        if(ii%2==0):
            I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,i), self.x[ii-1], self.x[ii+1])
            self.eI += eI
        else:
            I,eI = integrate.quad(lambda x: Kernel(x)*self.f(x,i), self.x[ii-2], self.x[ii+2])
            self.eI += eI
        return I
    
    def ComputeIJ(self,i,j,Kernel):
        # I need to fix the i=0 case
        ii=i+1
        jj=j+1
        if(ii%2==0):
            xx = (self.x[ii-1], self.x[ii+1])
        else:
            xx = (self.x[ii-2], self.x[ii+2])
        if(jj%2==0):
            yy = (self.x[jj-1], self.x[jj+1])
        else:
            yy = (self.x[jj-2], self.x[jj+2])
        
        I,eI = integrate.dblquad(lambda x,y: self.f(x,i)*Kernel(x,y)*self.f(y,j), yy[0], yy[1],xx[0], xx[1])
        self.eI += eI

        return I

def interp(x,q,fe):
    S = 0*x
    for k in range(fe.N):
        S+= fe.f(x,k)*q[k]
    return S


#### MODELS ####


class simple_PDF():
    def __init__(self,a,b,g): 
        self.a=a
        self.b=b
        self.g=g
        self.r = 1.0
        self.F = lambda y: (y**a*(1-y)**b*(1 + g*np.sqrt(y)))/self.r
        self.r,e = integrate.quad(self.F,0.0,1.0)  


def DPDFnormed(x,a,b):
    P=tr.tensor([a,b])
    a,b=P[0],P[1]
    dG_da,dG_db=dNorm(P)
    N=tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))
    dP_da=(tr.pow(x,a))*tr.pow(1-x,b)*tr.log(x) *N+dG_da*x**a*(1-x)**b
    dP_db= (tr.pow(x,a))*tr.pow(1-x,b)*tr.log(1-x) *N + dG_db*x**a*(1-x)**b
    return dP_da,dP_db

def Normalization(P):
    a,b=P[0],P[1]
    return tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))

def dNorm(P):
    a,b=P[0],P[1]
    dG_da= tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))*(tr.digamma(a+b+2) - tr.digamma(a+1))
    dG_db= tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))*(tr.digamma(a+b+2) - tr.digamma(b+1))
    return tr.tensor([dG_da,dG_db])


def simplePDFnormed(x,a,b):
    return tr.pow(x,a)*tr.pow(1-x,b)*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))
#x**a*(1-x)**b*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))


def very_simplePDFnormed(x,b):
    return (1-x)**b*tr.exp(gammaln(b+2) - gammaln(b+1))

# Posterior GP V2 with split RBF kernel
def pseudo_data(nu,a,b,g,da,db,dg,N):
    sa = np.random.normal(a,da,N)
    sb = np.random.normal(b,db,N)
    sg = np.random.normal(g,dg,N)

    D = np.zeros((N,nu.shape[0]))
    Norm=1.0
    for k in range(N):
        for i in range(nu.shape[0]):
            F =  lambda y: y**sa[k]*(1-y)**sb[k]*(1 + sg[k]*np.sqrt(y)-0.1*y)*np.cos(nu[i]*y) 
            r,e = integrate.quad(F,0.0,1.0) 
            D[k,i] = r
            if i==0:
                Norm = r
            D[k,i] = D[k,i]/Norm
    #add additional gaussian noise to break correlations
    NN = np.random.normal(0,1e-2,np.prod(D.shape)).reshape(D.shape)
    return D+NN

def autograd(func,x):
    x_tensor = x.clone().detach()
    x_tensor.requires_grad_()
    y = func(x_tensor)
    y.backward()
    return x_tensor.grad

def DPDFnormed(x,a,b):
    P=tr.tensor([a,b])
    a,b=P[0],P[1]
    dG_da,dG_db=dNorm(P)
    N=tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))
    dP_da=(tr.pow(x,a))*tr.pow(1-x,b)*tr.log(x) *N+dG_da*x**a*(1-x)**b
    dP_db= (tr.pow(x,a))*tr.pow(1-x,b)*tr.log(1-x) *N + dG_db*x**a*(1-x)**b
    return dP_da,dP_db

def Normalization(P):
    a,b=P[0],P[1]
    return tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))

def dNorm(P):
    a,b=P[0],P[1]
    dG_da= tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))*(tr.digamma(a+b+2) - tr.digamma(a+1))
    dG_db= tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))*(tr.digamma(a+b+2) - tr.digamma(b+1))
    return tr.tensor([dG_da,dG_db])


def simplePDFnormed(x,a,b,N):
    return N*tr.pow(x,a)*tr.pow(1-x,b)*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))

#xtensor=tr.tensor(x_grid)
def model(x):
    a=x[0]
    b=x[1]
    xtensor=tr.tensor([0.5])
    return simplePDFnormed(xtensor,a,b)

#### Kernels #####

def KrbfMat(x,s,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*s*tr.exp(-0.5*((xx - yy)/w)**2)

def Krbf_no_s(x,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return tr.exp(-0.5*((xx - yy)/w)**2)

def Krbf1(x,s,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    w=10**w
    s=10**s
    return s**2*tr.exp(-0.5*((xx - yy)/w)**2)

class splitRBFker():
    def __init__(self,sp,scale=1):
        self.sp =sp
        self.scale = scale
    def KerMat(self,x,s1,w1,s2,w2):
        K2 = KrbfMat(x,s2,w2) # linear
        K1 = KrbfMat(tr.log(x),s1,w1)
        sig = tr.diag(tr.special.expit(self.scale*(x-self.sp)))
        sigC = tr.eye(x.shape[0])-sig
        ##return K1+K2
        return sigC@K2@sigC + sig@K1@sig

def Sig(x,scale,sp=0.1):
    return tr.special.expit(scale*(x-sp))
def transform(s):
    return s.view(s.shape[1],1).repeat(1,s.shape[1])

#  write the last one as a function
def splitRBFkerMat(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = KrbfMat(tr.log(x+eps),s2,w2) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

#DERIVATIVES
def Krbf_ds(x,s,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return 2*s*tr.exp(-0.5*((xx - yy)/w)**2)
    #return  2*s*tr.exp(-0.5*((x.view(1,x.shape[0]) - x.view(x.shape[0],1))/w)**2)
def Krbf_dw(x,s,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*s*tr.exp(-0.5*((xx - yy)/w)**2)*(xx-yy)**2/((w**3))

def sig_ds(x,scale,sp=0.1):
    sig = tr.special.expit(scale*(x.view(1,x.shape[0])-sp))
    return sig*(1-sig)

def Kcom_ds1(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    sig = tr.special.expit(scale*(x.view(1,x.shape[0])-sp))
    sig=sig.view(sig.shape[1],1).repeat(1,sig.shape[1])
    #sigC = 1-sig
    return sig*Krbf_ds(x,s1,w1)*sig.T
def Kcom_dw1(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    sig = tr.special.expit(scale*(x.view(1,x.shape[0])-sp))
    sig=sig.view(sig.shape[1],1).repeat(1,sig.shape[1])
    #sigC = 1-sig
    return sig*Krbf_dw(x,s1,w1)*sig.T
def Kcom_ds2(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-15):
    sig = tr.special.expit(scale*(x.view(1,x.shape[0])-sp))
    sig=sig.view(sig.shape[1],1).repeat(1,sig.shape[1])
    sigC = 1-sig
    return sigC*Krbf_ds(tr.log(x+eps),s2,w2)*sigC.T
def Kcom_dw2(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    sig = tr.special.expit(scale*(x.view(1,x.shape[0])-sp))
    sig=sig.view(sig.shape[1],1).repeat(1,sig.shape[1])
    sigC = 1-sig
    return sigC*Krbf_dw(tr.log(x+eps),s2,w2)*sigC.T

def sig_ds(x,scale,sp=0.1):
    return tr.exp(-scale*(x-sp))*(x-sp)*tr.special.expit(scale*(x-sp))**2

def Kcom_ds(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    K2=KrbfMat(tr.log(x+eps),s2,w2)
    K1=KrbfMat(x,s1,w1)
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    ##vectors
    ssx=Sig(xx,scale,sp)
    ssy=Sig(yy,scale,sp)
    #transform into matrix
    sx=transform(ssx)
    sy=transform(ssy.T)

    dssx=sig_ds(xx,scale,sp)
    dssy=sig_ds(yy,scale,sp)
    #transform into matrix
    dsx=transform(dssx)
    dsy=transform(dssy.T)

    F1=((-1+sy.T)*dsx + (sx-1)*dsy.T)*K2
    F2=((dsx)*sy.T + sx*(dsy.T))*K1
    return F1+F2

def R(z,t):
    return 1.0/tr.sqrt(1-2*z*t+t*t)

def jacobi(x,s,t,a,b):
   x=x.view(x.shape[0],1)
   y=x.view(1,x.shape[0])
   return (s**2)*(x*y)**a*((1-x)*(1-y))**b*(R(2*x-1,t)*R(2*y-1,t)*((1-t+R(2*x-1,t))*(1-t+R(2*y-1,t)))**a*((1+t+R(2*x-1,t))*(1+t+R(2*y-1,t)))**b)**(-1)#+1e-6*tr.eye(x.shape[0])
   #return s*(x.view(1,x.shape[0])*x.view(x.shape[0],1))**a*((1-x.view(1,x.shape[0]))*(1-x.view(x.shape[0],1)))**b*(R(2*x.view(1,x.shape[0])-1,t)*R(2*x.view(x.shape[0],1)-1,t)*((1-t+R(2*x.view(1,x.shape[0])-1,t))*(1-t+R(2*x.view(x.shape[0],1)-1,t)))**a*((1+t+R(2*x.view(1,x.shape[0])-1,t))*(1+t+R(2*x.view(x.shape[0],1)-1,t)))**b)**(-1)

#plot the trace1
def plotrace(trace,burn=100,kernel='jacobi'):
    fig, ax = plt.subplots(trace.shape[1],figsize=(20, 8))
    i0 = burn
    iF=trace.shape[0]
    if kernel=='jacobifull':
        lab=['α', 'β','s','t','a','b']
    elif kernel=='jacobi':
        lab=['s','t','a','b']
    else:
        lab=['α', 'β','σ1','w1','σ2','w2','s','σerror']
    col=['red','blue','green','pink','black','orange','purple','brown']
    for i in range(trace.shape[1]):
        ax[i].plot(trace[i0:iF,i],label=lab[i],color=col[i])
        ax[i].legend()
    plt.show()

def plothist(trace,mygp,disc,prior=False,burn=100,kernel='jacobi'):
    fig, ax = plt.subplots(trace.shape[1], 1, figsize=(10, 10), sharex=False, sharey=False)
    i0 = 100
    iF=10000
    if kernel=='jacobifull':
        lab=['α', 'β', 's', 't', 'a', 'b']
        labprior=['α-prior', 'β-prior', 's-prior', 't-prior', 'a-prior', 'b-prior']
    elif kernel=='jacobi':
        lab=['s','t','a','b']
        labprior=['s-prior','t-prior','a-prior','b-prior']
    else:
        lab=['α','β','σ1','w1','σ2','w2','s','σerror']
        labprior=['α-prior','β-prior','σ1-prior','w1-prior','σ2-prior','w2-prior','s-prior','σerror-prior']
    col=['red','blue','green','pink','black','orange','purple','brown','lime','cyan','magenta','yellow']


    for i in range(trace.shape[1]):
        ax[i].hist(trace[i0:iF,i],bins=disc,label=lab[i],color=col[i],density=True)
        if prior:
            initial=mygp.prior_dist[i].shift
            final=mygp.prior_dist[i].shift+mygp.prior_dist[i].scale
            xxx = tr.linspace(initial,final,1000)
            distexp=mygp.prior_dist[i]
            pdfs=tr.zeros(xxx.shape[0])
            for k in range(xxx.shape[0]):
                pdfs[k]=distexp.pdf(xxx[k])
            ax[i].plot(xxx,pdfs.detach().numpy())
            ax[i].set_xlim([initial-0.5,final+0.5])
        ax[i].legend()
    plt.show()


def RBF(x,s,w):
    return s*s*tr.exp(-0.5*((x.view(1,x.shape[0]) - x.view(x.shape[0],1))/w)**2)


#from tensor to list
def tensor2list(tensor):
    return [tensor[i].item() for i in range(tensor.shape[0])]

Nx=256
x_grid = np.concatenate((np.logspace(-12,-1,np.int32(Nx/2)),np.linspace(0.1+1e-4,1-1e-12,np.int32(Nx/2))))

#mockdata


#set up input data
def preparedata(i,nu,rMj,rMe,rM,scale="log"):
    #prepare the data
    Nnu = nu.shape[1]
    CovD= np.corrcoef(rMj[:,:,i-1].T)#*(rMj[:,:,i-1].T.shape[0]-1)
    CovD =tr.tensor( (CovD + CovD.T)/2)
    M = rM.T[i]
    eM = rMe.T[i]
    n = nu.T[i]
    Nx=256
    if scale=="lin":
        x_grid = np.linspace(0.0+1e-12,1-1e-12,np.int32(Nx))
    elif scale=="log":
        x_grid = np.concatenate((np.logspace(-12,-1,np.int32(Nx/2)),np.linspace(0.1+1e-4,1-1e-12,np.int32(Nx/2))))
    #x_grid = np.concatenate((np.logspace(-12,-1,np.int32(Nx/2)),np.linspace(0.1+1e-4,1-1e-12,np.int32(Nx/2))))
    fe = FE2_Integrator(x_grid)
    # soften the constrants
    lam = 1e-7  #normalization
    lam_c = 1e-5 #x=1 
    B0 = fe.set_up_integration(Kernel=lambda x: 1)
    B1 = np.zeros_like(B0) 
    B1[-1] = 1.0 # x=1 is at the end...
    n # is the nu values at current z
    B = np.zeros((n.shape[0],Nx))
    for k in np.arange(n.shape[0]):
        B[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(n[k]*x))
    V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:],B))
    Gamma = np.zeros((V.shape[0],V.shape[0]))
    Gamma[0,0] = lam
    Gamma[1,1] = lam_c
    Gamma[2:,2:] = CovD
    Y = np.concatenate(([1,0],M))
    return x_grid,V,Y,Gamma

def preparemockdata(Nnupoints):
    #MOCK data
    #######Generate mock data to test the GP

    numock = np.linspace(0,13,Nnupoints)
    #create fake data
    #a,b,c
    jM = pseudo_data(numock,-0.3,2.9,1.0,0.02,.2,0.2,1000)
    #print(jM.shape)
    M = np.mean(jM,axis=0)
    eM = np.std(jM,axis=0)
    """print("Check the zero point:",nu[0],M[0],eM[0])
    plt.errorbar(numock,M,eM,marker='o')
    plt.show()"""
    #chop off the nu = 0
    jM = jM[:,1:]
    n = numock[1:]
    M = np.mean(jM,axis=0)
    eM = np.std(jM,axis=0)

    
    #print("jM shape: ",jM.shape)

    CovD = np.corrcoef(jM.T)   
    CovD =(CovD + CovD.T)/2.0
    """U,S,V = np.linalg.svd(CovD)
    #print("Data Cov: ",CovD)
    print("Data Cov S:",S)
    #plot covD
    plt.imshow(CovD)
    plt.show()"""

    Nx=256
    x_grid = np.concatenate((np.logspace(-12,-1,np.int32(Nx/2)),np.linspace(0.1+1e-4,1,np.int32(Nx/2))))
    fe = FE2_Integrator(x_grid)
    lam = 1e-5   # soften the constrants
    lam_c = 1e-5
    B0 = fe.set_up_integration(Kernel=lambda x: 1)
    B1 = np.zeros_like(B0) 
    B1[-1] = 1.0 # x=1 is at the end...
    n # is the nu values at current z
    B = np.zeros((n.shape[0],Nx))
    for k in np.arange(n.shape[0]):
        B[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(n[k]*x))
    V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:],B))
    Gamma = np.zeros((V.shape[0],V.shape[0]))
    Gamma[0,0] = lam
    Gamma[1,1] = lam_c
    Gamma[2:,2:] = CovD
    Y = np.concatenate(([1,0],M))
    return x_grid,V,Y,Gamma

fits_comb=[]
sss=tr.tensor([1.0,0.2,1.0,5.1,1.0,2.2,1.0])
mmm=tr.tensor([-0.5,1.5,1.0,.1,1.0,2.2,1.0])
print("0=gaussian, 1=lognormal, 2=expbeta")
for i in range(0,nu.shape[1]):
    x_gri0,V0,Y0,Gamma0 = preparedata(i,nu,rMj,rMe,rM)
    myGP0= GaussianProcess(x_gri0,V0,Y0,Gamma0,f"z={i+1}a",flag="noisy",device="cpu",Pd=simplePDFnormed, Ker=splitRBFkerMat,Pd_args=(-0.0,1.0,1.0),Ker_args=(10.0,0.1,10.0,0.1,0.1,0.1))
    myGP0.prior2ndlevel("all",1000,mean=tr.tensor([-1.0,0.0,0.0,0.00,0.0,0.0,0.0,0.0,0.0]),sigma=tr.tensor([3.0,5.0,6.0,11.0,6.0,11.0,6.0,2.0,2.0]),prior_mode=tr.tensor([2,2,2,2,2,2,2,2,2]))
    fits_comb.append(myGP0)
    print(fits_comb[i].name, "done")
x_gri0,V0,Y0,Gamma0 = preparemockdata(7)
myGP0= GaussianProcess(x_gri0,V0,Y0,Gamma0,f"z=mock",flag="noisy",device="cpu",Pd=simplePDFnormed, Ker=splitRBFkerMat,Pd_args=(-0.0,1.0,1.0),Ker_args=(10.0,0.1,10.0,0.1,0.1,0.1))
myGP0.prior2ndlevel("all",1000,mean=tr.tensor([-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]),sigma=tr.tensor([3.0,5.0,6.0,11.0,6.0,11.0,5.5,2.0,2.0]),prior_mode=tr.tensor([2,2,2,2,2,2,2,2,2]))
fits_comb.append(myGP0)
print(fits_comb[-1].name, "done")

mode='all'
Ntrain=1000
function="nlp"

i=args.i

if i in [1,2,3]:
    fits_comb[i].train(Ntrain,lr=1e-2,mode=mode,function="nlp")
elif i in [12]:
    fits_comb[i].train(Ntrain,lr=1e-2,mode=mode,function="nlp")
else:
    fits_comb[i].train(Ntrain,lr=1e-2,mode=mode,function="nlp")
print(tr.tensor(fits_comb[i].pd_args +fits_comb[i].ker_args  + (fits_comb[i].sig,)))


##sample kernel
momentum=1*tr.tensor([1.0,1.0,1.0,1.0,1.0,1.0])#,1.0,2.0,1.0])
sigma=tr.tensor([3.0,5.0,6.0,5.0,6.0,5.0,5.5,2.0])
samplers=[]
for i in range(0,13):
    #GPsampler=HMC_sampler(myGP0.nlogpost2levelpdf,grad=myGP0.gradlogpost2levelpdf,diagonal=momentum)
    #shift=tr.tensor(fits_comb[i].pd_args + fits_comb[i].ker_args  + (fits_comb[i].sig,)).to("cpu")-tr.ones(8)
    #shift=tr.tensor([0,0, 0.9851, 1.7835, 1.6409, 0.8482, 1.0654,0.0627])-tr.ones(8)
    #fits_comb[i].prior2ndlevel("kernel",1000,mean=shift,sigma=2*tr.ones(8),prior_mode=tr.tensor([2,2,2,2,2,2,2,2]))
    GPsampler=HMC_sampler(fits_comb[i].nlogpost2levelpdf,device="cpu",diagonal=1.0*tr.ones(fits_comb[i].Nparams),grad=None)
    #rand=tr.rand(7)*5
    
    #GPsampler.q0=tr.tensor(fits_comb[i].ker_args  + (fits_comb[i].sig,)).to("cpu")
    GPsampler.q0=tr.tensor(fits_comb[i].pd_args + fits_comb[i].ker_args+ (fits_comb[i].sig,)).to("cpu")
    #GPsampler.q0=tr.tensor(fits_comb[i].pd_args + fits_comb[i].ker_args).to("cpu")+0.02#+ (fits_comb[i].sig,))
    samplers.append(GPsampler)
    print(fits_comb[i].name,"sampler done")

i=args.i
Nsamples=args.Nsamples
burn=args.burn
L=args.L
eps=args.eps

traceq,tracep,traceH=samplers[i].sample(samplers[i].q0,Nsamples,eps,L)#,update=1)
tr.save(traceq,'Krbf+log1(%s).pt' %(fits_comb[i].name))
