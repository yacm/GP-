import numpy as np 
import torch as tr
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import scipy.integrate as integrate
from torch.special import gammaln
from torch.autograd.functional import hessian
import scipy.integrate as integrate
import h5py as h5
from scipy.optimize import minimize 
from scipy import special 
from scipy.optimize import Bounds 
from scipy.linalg import cho_solve 
import time
from sklearn.preprocessing import MinMaxScaler 
from sklearn.pipeline import Pipeline 
import scipy.special

def get_dist_matelem(z, p, t_min,ITD="Re"):
    f = 0
    if p <= 3:
        f = h5.File('pdf-data/Nf2+1/ratio.summationLinearFits.cl21_32_64_b6p3_m0p2350_m0p2050.unphased.hdf5','r')
    else:
        f = h5.File('pdf-data/Nf2+1/ratio.summationLinearFits.cl21_32_64_b6p3_m0p2350_m0p2050.phased-d001_2.00.hdf5','r')
    M_z_p = np.array(f['MatElem/bins/'+ITD+'/mom_0_0_+'+str(p)+'/disp_z+'+str(z)+'/insertion_gt/tsep_'+str(t_min)+'-14'])
    M_0_0 = np.array(f['MatElem/bins/Re/mom_0_0_0/disp_0/insertion_gt/tsep_'+str(t_min)+'-14'])
    M_z_0 = np.array(f['MatElem/bins/Re/mom_0_0_0/disp_z+'+str(z)+'/insertion_gt/tsep_'+str(t_min)+'-14'])
    M_0_p = np.array(f['MatElem/bins/Re/mom_0_0_+'+str(p)+'/disp_0/insertion_gt/tsep_'+str(t_min)+'-14'])
    
    f.close()
    return M_z_p * M_0_0 / M_0_p / M_z_0

def get_final_res(z, p,ITD):
    m_4, _ = get_dist_matelem(z, p, 4,ITD)
    m_6, s_6 = get_dist_matelem(z, p, 6,ITD)
    m_8, s_8 = get_dist_matelem(z, p, 8,ITD)
    return m_6, np.sqrt(s_6**2)#+(m_4-m_6)**2)

def get_data(ITD):
    Np = 6
    Nz = 12
    Nj = 349
    rMj = np.empty([Nj,Np,Nz])
    nu = np.empty([Np,Nz])
    for p in range(1,Np+1):
        for z in range (1,Nz+1):
            nu[p-1,z-1] = 2.0*np.pi/32.0 *p *z
            #print(p,z,nu[p-1,z-1])
            m_4 = get_dist_matelem(z,p,4,ITD)
            m_6 = get_dist_matelem(z,p,6,ITD)
            m_8 = get_dist_matelem(z,p,8,ITD)
            #expo fit
            m = (m_4*m_8 - m_6**2)/(m_4 + m_8 - 2 * m_6)
            # this fails for certain cases where the denomenator goes too close to zero
            # use the m_6 as default
            rMj[:,p-1,z-1] = m_6
            #Nj=m.shape[0]
            #print(z,p,np.mean(m_4),np.mean(m_6),np.mean(m_8), np.mean(m),np.std(m)*np.sqrt(Nj-1))
    rM = np.mean(rMj,axis=0)
    rMe = np.std(rMj,axis=0)*np.sqrt(Nj) 
    return nu,rMj,rMe,rM

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


def PDFnormed(x,a,b):
    return tr.pow(x,a)*tr.pow(1-x,b)*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))
#x**a*(1-x)**b*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))


def very_simplePDFnormed(x,b):
    return (1-x)**b*tr.exp(gammaln(b+2) - gammaln(b+1))

# Posterior GP V2 with split RBF kernel
# Posterior GP V2 with split RBF kernel
def pseudo_data(nu,a,b,g,da,db,dg,N,ITD="Re",Model="PDF"):

    sa = np.random.normal(a,da,N)
    sb = np.random.normal(b,db,N)
    sg = np.random.normal(g,dg,N)

    D = np.zeros((N,nu.shape[0]))
    Norm=1.0
    for k in range(N):
        for i in range(nu.shape[0]):
            if ITD=="Re":
                F =  lambda y: y**sa[k]*(1-y)**sb[k]*(1 + sg[k]*np.sqrt(y)-0.1*y)*np.cos(nu[i]*y) 
            else:
                F =  lambda y: y**sa[k]*(1-y)**sb[k]*(1 + sg[k]*np.sqrt(y)-0.1*y)*np.sin(nu[i]*y)
            r,e = integrate.quad(F,0.0000001,1.0-0.0000001) 
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


def PDF_N(x,a,b,N):
    return N*tr.pow(x,a)*tr.pow(1-x,b)*tr.exp(gammaln(a+b+2) - gammaln(a+1) - gammaln(b+1))

#xtensor=tr.tensor(x_grid)
def model(x):
    a=x[0]
    b=x[1]
    xtensor=tr.tensor([0.5])
    return PDF_N(xtensor,a,b)

def ModelC(x,N):
    return x-x+N

def PDF(x,a,b,N):
    return N*tr.pow(x,a)*tr.pow(1-x,b)

#### Kernels #####

def KrbfMat(x,s,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*s*tr.exp(-0.5*((xx - yy)/w)**2)

def Krbflog(x,s,w,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*s*tr.exp(-0.5*((tr.log(xx+eps) - tr.log(yy+eps))/w)**2)

def Krbflog_no_s(x,w,eps=1e-13):
    s=2.5**0.5
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*s*tr.exp(-0.5*((tr.log(xx+eps) - tr.log(yy+eps))/w)**2)

def Krbf_no_s(x,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return tr.exp(-0.5*((xx - yy)/w)**2)


def Krbf_fast(x,s,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    w=10**w
    s=10**s
    return s**2*tr.exp(-0.5*((xx - yy)/w)**2)

def Kpoly(x,s,t,a,b):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s**2*((xx*yy)**a*((1-xx)*(1-yy))**b)/((1-t*xx)*(1-t*yy))

def Kpoly2(x,s,a,b):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s**2*((xx*yy)**a*((1-xx)*(1-yy))**b)/((1-yy*xx))

def log_jac(x,s,t,a,b,s1,w1,scale,sp=0.1,eps=1e-12):
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K2 = KrbfMat(tr.log(x+eps),s1,w1) #log # linear
    K1 = jacobi(x,s,t,a,b) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def l(x,l0,eps=1e-10):
    return l0*(x+eps)

def Kdebbio(x,sig,l0,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return sig**2*tr.sqrt(2*l(xx,l0,eps)*l(yy,l0,eps)/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*tr.exp(-(xx-yy)**2/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))



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
def rbf_logrbf(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
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

def rbf_logrbf_s1(x,s1,w1,s2,w2,scale=1.0,sp=0.1,eps=1e-12):
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


#  write the last one as a function
def rbf_logrbf_s_w(x,s1,s2,scale=1,sp=0.1,eps=1e-14):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,s1) # linear
    K2 = KrbfMat(tr.log(x+eps),s2,s2) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def rbf_deb(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = Kdebbio(x,s2,w2,eps) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def rbf_deb_s1(x,s1,w1,s2,w2,scale=0.1,sp=0.1,eps=1e-12):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = Kdebbio(x,s2,w2,eps) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def rbf_deb_s_w(x,s1,s2,scale=1,sp=0.1,eps=1e-13):
    K1 = KrbfMat(x,s1,s1) # linear
    K2 = Kdebbio(x,s2,s2,eps=eps) #log
    xx=x.view(1,x.shape[0])
    ss=Sig(xx,scale,sp)
    s=transform(ss)
    #sig=sig.view(1,sig.shape[1]).repeat(sig.shape[1],1)
    sC = 1-s
    return  s*K1*s.T +sC*K2*sC.T

def splitRBF1(x,s1,w1,s2,w2,scale,sp=0.1,eps=1e-12):
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

def l(x,l0,eps=1e-10):
    return l0*(x+eps)


def Kdebbio(x,l0,sig,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return sig**2*tr.sqrt(2*l(xx,l0,eps)*l(yy,l0,eps)/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*tr.exp(-(xx-yy)**2/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))

def Kdebbioxa(x,l0,sig,a,eps=1e-12):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return xx**a*sig**2*tr.sqrt(2*l(xx,l0,eps)*l(yy,l0,eps)/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*tr.exp(-(xx-yy)**2/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*yy**a

def Kdebbioxb(x,l0,sig,b,eps=1e-13):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return (1-xx)**b*sig**2*tr.sqrt(2*l(xx,l0,eps)*l(yy,l0,eps)/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*tr.exp(-(xx-yy)**2/(l(xx,l0,eps)**2+l(yy,l0,eps)**2))*(1-yy)**b

def KSM(x,s1,l1,m1,s2,l2,m2):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s1*s1*tr.exp(-0.5*((xx - yy)**2/l1**2))*tr.cos(m1*(xx-yy)**2)+s2*s2*tr.exp(-0.5*((xx - yy)**2/l2**2))*tr.cos(m2*(xx-yy)**2)

def R(z,t):
    return tr.sqrt(1-2*z*t+t*t)

def F(z,t,a,b):
    return 1/(R(z,t)*(1-t+R(z,t))**a*(1+t+R(z,t))**b)

def jacobi(x,s,t,a,b):
   x=x.view(x.shape[0],1)
   y=x.view(1,x.shape[0])
   return (s**2)*(x*y)**a*((1-x)*(1-y))**b* F(2*x-1,t,a,b)* F(2*y-1,t,a,b)

def rbf_logrbf_no_s(x,w1,w2,s1=5.0,s2=10.0,scale=5.0,sp=0.1,eps=1e-12):
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

def rbf_deb_no_s(x,w1,w2,s1=1.0,s2=1.0,scale=1.0,sp=0.1,eps=1e-12):
    #plot this values and it looks like a simple rbf kernel
    #s1,w1,s2,w2,scale,sp =  1.0,0.1,1.0,2.2,1.0,.1
    K1 = KrbfMat(x,s1,w1) # linear
    K2 = Kdebbio(x,s2,w2) #log
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

#set up input data
def preparedata(i,nu,rMj,rMe,rM,x_grid,ITD="Re"):


    #prepare the data
    Nx = x_grid.shape[0]
    #CovD= np.corrcoef(rMj[:,:,i-1].T)#*(rMj[:,:,i-1].T.shape[0]-1)

    Nj = 349 #data points

    """ Np = 6
    Nz = 12
    Nj = 349
    rMj = np.empty([Nj,Np,Nz])"""
    #indices [p,z,
    CovD= np.cov(rMj[:,:,i].T*np.sqrt(Nj-1))
    CovD=(CovD+CovD.T)/2.0
    #CovD=CovD**0.5
    #change nans by 0
    CovD=np.abs(CovD)
    CovD[CovD<0]=0

    M = rM.T[i]
    eM = rMe.T[i]
    n = nu.T[i]
    
    fe = FE2_Integrator(x_grid)
    # soften the constrants
 
    B0 = fe.set_up_integration(Kernel=lambda x: 1)
    B1 = np.zeros_like(B0) 
    B1[-1] = 1.0 # x=1 is at the end...
    n # is the nu values at current z
    B = np.zeros((n.shape[0],Nx))
    for k in np.arange(n.shape[0]):
        if ITD=="Re":
            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(n[k]*x))
            lam = 1e-10  #normalization
            lam_c = 1e-10 #x=1 

        elif ITD=="Im":
            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.sin(n[k]*x))
            #lam = 1e5  #normalization
            lam_c = 1e-10 #x=1 
            

    if ITD=="Re":
        V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam
        Gamma[1,1] = lam_c
        if i<-1:
            Gamma[2:,2:] = np.diag(eM)
        else:
            Gamma[2:,2:] = CovD#np.diag(eM)#CovD
        Y = np.concatenate(([1.0,0.0],M))
    elif ITD=="Im":
        V = np.concatenate((B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        #Gamma[0,0] = lam
        Gamma[0,0] = lam_c
        Gamma[1:,1:] = CovD#np.diag(eM)#CovD
        Y = np.concatenate(([0.0],M))

    return x_grid,V,Y,Gamma

#gamma function numpy
import math
def gamma(x):
    return math.gamma(x)
def PDF_np(x,a,b):
    return x**a*(1-x)**b*gamma(a+b+2)/(gamma(a+1)*gamma(b+1))
def mockpdf(xgrid,a,b,da,db,N):

    pdf = np.zeros((xgrid.shape[0],N))
    sa = np.random.normal(a,da,N)
    sb = np.random.normal(b,db,N)
    for i in range(xgrid.shape[0]):
        for j in range(N):
            pdf[i,j] = PDF_np(xgrid[i],sa[j],sb[j])
    return pdf

def pseudo_data1(nu_grid,x_grid,a,b,da,db,ITD="Re"):
    #generate data
    fe=FE2_Integrator(x_grid)
    BB = np.zeros((nu_grid.shape[0],x_grid.shape[0]))
    for k in np.arange(nu_grid.shape[0]):
        if ITD=="Re":
            BB[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(nu_grid[k]*x))
        elif ITD=="Im":
            BB[k,:] = fe.set_up_integration(Kernel= lambda x : np.sin(nu_grid[k]*x))
    pdf=mockpdf(x_grid,a,b,da,db,1000)
    return pdf.T @ BB.T

def preparemockdata1(Nnupoints,numax,x_grid,ITD="Re"):
    nu_grid = np.linspace(0,numax,Nnupoints)
    Nx=x_grid.shape[0]
    nu_grid=nu_grid[1:]
    if ITD=="Re":
        if numax==25:
            Reg=1e-11*np.identity(nu_grid.shape[0])
        elif numax==10:
            Reg=1e-12*np.identity(nu_grid.shape[0])
        elif numax==4:
            Reg=1e-13*np.identity(nu_grid.shape[0])
    elif ITD=="Im":
        if numax==25:
            Reg=1e-8*np.identity(nu_grid.shape[0])
        elif numax==10:
            Reg=1e-10*np.identity(nu_grid.shape[0])
        elif numax==4:
            Reg=1e-13*np.identity(nu_grid.shape[0])

    fe = FE2_Integrator(x_grid)

    #generate data
    itd=pseudo_data1(nu_grid,x_grid,-0.2,2.5,0.1,0.5,ITD=ITD)
    M=itd.mean(axis=0)
    CovD= np.cov(itd.T)

    B0 = fe.set_up_integration(Kernel=lambda z: 1)
    B1 = np.zeros_like(B0) 
    B1[-1] = 1.0 # x=1 is at the end... #Delta function
    # is the nu values at current z
    B = np.zeros((nu_grid.shape[0],Nx))
    for k in np.arange(nu_grid.shape[0]):
        if ITD=="Re":
            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(nu_grid[k]*x))
            lam = 1e-10   # soften the constrants
            lam_c = 1e-10
        elif ITD=="Im":

            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.sin(nu_grid[k]*x))
            lam_c = 1e-10
    if ITD=="Re":
        V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam
        Gamma[1,1] = lam_c
        Gamma[2:,2:] = CovD + Reg #np.diag(eM)#CovD
        Y = np.concatenate(([1.0,0.0],M))
    elif ITD=="Im":
        V = np.concatenate((B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam_c
        Gamma[1:,1:] = CovD#np.diag(eM**2)
        Y = np.concatenate(([0.0],M))

    return x_grid,V,Y,Gamma

def NNPDFdata(datanu,x_grid,regulator=True,ITD="Re"):
    nu_d_grid = datanu.T[1]
    numax=nu_d_grid.shape[0]
    Nx=x_grid.shape[0]
    if regulator:
        if ITD=="Re":
            if numax==25:
                Reg=1e-8*np.identity(nu_d_grid.shape[0])
            elif numax==10:
                Reg=1e-10*np.identity(nu_d_grid.shape[0])
            elif numax==4:
                Reg=1e-10*np.identity(nu_d_grid.shape[0])
        elif ITD=="Im":
            if numax==25:
                Reg=1e-7*np.identity(nu_d_grid.shape[0])
            elif numax==10:
                Reg=1e-7*np.identity(nu_d_grid.shape[0])
            elif numax==4:
                Reg=1e-7*np.identity(nu_d_grid.shape[0])
    else:
        Reg=0


    M=datanu.T[2:].mean(axis=0)
    eMnu=datanu.T[2:].std(axis=0)*np.sqrt(datanu.shape[0])
    CovD=np.cov(datanu.T[2:].T)
    #Symetrize the matrix CovD
    #print("Symetrize the matrix CovD")
    CovD=(CovD+CovD.T)/2.0
    

    fe = FE2_Integrator(x_grid)
    B0 = fe.set_up_integration(Kernel=lambda z: 1)
    B1 = np.zeros_like(B0) 
    B1[-1] = 1.0 # x=1 is at the end... #Delta function
    # is the nu values at current z
    B = np.zeros((nu_d_grid.shape[0],Nx))
    for k in np.arange(nu_d_grid.shape[0]):
        if ITD=="Re":
            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(nu_d_grid[k]*x))
            lam = 1e-10   # soften the constrants
            lam_c = 1e-10
        elif ITD=="Im":

            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.sin(nu_d_grid[k]*x))
            lam_c = 1e-10
    if ITD=="Re":
        V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam
        Gamma[1,1] = lam_c
        #Gamma[2:,2:] = CovD + Reg#np.diag(eM)#CovD
        if numax>6:
            #print("Flag1")
            Gamma[2:,2:] = np.diag(np.diag(CovD))
        else:
            Gamma[2:,2:] = CovD +Reg#np.diag(eM**2)
        Y = np.concatenate(([1.0,0.0],M))
    elif ITD=="Im":
        V = np.concatenate((B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam_c
        if numax>6:
            #print("Flag1")
            Gamma[1:,1:] = np.diag(np.diag(CovD))
        else:
            Gamma[1:,1:] = CovD +Reg#np.diag(eM**2)
        Y = np.concatenate(([0.0],M))

    return x_grid,V,Y,Gamma


def preparemockdata(Nnupoints,numax,ITD="Re",Nx=256):
    #MOCK data
    #######Generate mock data to test the GP

    numock = np.linspace(0,numax,Nnupoints)
    #create fake data
    if numax==25:
        Reg=1e-11*np.identity(Nnupoints)
    elif numax==10:
        Reg=1e-12*np.identity(Nnupoints)
    elif numax==4:
        Reg=1e-13*np.identity(Nnupoints)
    #a,b,c
    jM = pseudo_data(numock,-0.2,3.0,0.01,0.2,0.2,0.01,1000,ITD=ITD)
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

    #CovD = np.corrcoef(jM.T)   

    CovD= np.cov(jM.T)
    CovD=(CovD+CovD.T)/2
    #CovD=CovD**0.5
    #change nans by 0
    #CovD=np.abs(CovD)
    CovD[np.isnan(CovD)]=0

    """U,S,V = np.linalg.svd(CovD)
    #print("Data Cov: ",CovD)
    print("Data Cov S:",S)
    #plot covD
    plt.imshow(CovD)
    plt.show()"""

    x_grid = np.concatenate((np.logspace(-12,-1,np.int32(Nx/2)),np.linspace(0.1+1e-4,1-1e-12,np.int32(Nx/2))))
    fe = FE2_Integrator(x_grid)

    B0 = fe.set_up_integration(Kernel=lambda z: 1)
    B1 = np.zeros_like(B0) 
    B1[-1] = 1.0 # x=1 is at the end... #Delta function
    # is the nu values at current z
    B = np.zeros((n.shape[0],Nx))
    for k in np.arange(n.shape[0]):
        if ITD=="Re":
            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.cos(n[k]*x))
            lam = 1e-10   # soften the constrants
            lam_c = 1e-10
        elif ITD=="Im":

            B[k,:] = fe.set_up_integration(Kernel= lambda x : np.sin(n[k]*x))
            lam_c = 1e-10
            
    if ITD=="Re":
        V = np.concatenate((B0[np.newaxis,:],B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam
        Gamma[1,1] = lam_c
        Gamma[2:,2:] = CovD + Reg #np.diag(eM)#CovD
        Y = np.concatenate(([1.0,0.0],M))
    elif ITD=="Im":
        V = np.concatenate((B1[np.newaxis,:],B))
        Gamma = np.zeros((V.shape[0],V.shape[0]))
        Gamma[0,0] = lam_c
        Gamma[1:,1:] = CovD#np.diag(eM**2)
        Y = np.concatenate(([0.0],M))
    return x_grid,V,Y,Gamma


def arguments(modelname,kernelname,nugget,device,mode,ID):
    if modelname==PDF_N.__name__:
        meanf=tr.tensor([-1.0,0.0,0.0])
        sigmaf=tr.tensor([2.0,6.0,4.0])
        configf=tr.tensor([2,2,2])
        mod=(-0.0,1.0,1.0)
        modfunc=PDF_N

    elif modelname==ModelC.__name__:#Constant model
        meanf=tr.tensor([0.0])
        sigmaf=tr.tensor([20.0])
        configf=tr.tensor([2])
        mod=(1.0,)
        modfunc=ModelC

    elif modelname==PDFnormed.__name__:
        meanf=tr.tensor([-1.0,0.0])
        sigmaf=tr.tensor([2.0,6.0])
        configf=tr.tensor([2,2])
        mod=(-0.0,1.0)
        modfunc=PDFnormed

    elif modelname==PDF.__name__:
        meanf=tr.tensor([-1.0,0.0,0.0])
        sigmaf=tr.tensor([2., 15., 15.])#2.0,7.0,20.0])
        configf=tr.tensor([2,2,2])
        mod=(0.0,1.0,2.0)
        modfunc=PDF

    #select the kernel
    if kernelname==rbf_logrbf.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([20., 10., 20., 10.,  2.]) #ModelC
        #sigmak=tr.tensor([11.0,6.0,11.0,6.0,2.0])
        configk=tr.tensor([2,2,2,2,2,2])
        #ker=(50.0,1.1,50.0,1.0,1.0)
        ker=(2.5,0.1,2.5,0.1,1.0)
        kerfunc=rbf_logrbf

    elif kernelname==rbf_deb.__name__: 
        meank=tr.tensor([0.0,0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([11.0,6.0,11.0,11.0,2.0])
        configk=tr.tensor([2,2,2,2,2,2])
        ker=(4.0,5.0,4.0,2.0,0.1)
        kerfunc=rbf_deb

    if kernelname==rbf_logrbf_s1.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([20.0,20.0,20.0,20.0])
        configk=tr.tensor([2,2,2,2])
        #ker=(50.0,1.1,50.0,1.0,1.0)
        ker=(5.0,0.1,5.0,1.0)
        kerfunc=rbf_logrbf_s1

    if kernelname==rbf_logrbf_s_w.__name__:
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([20.0,20.0])
        configk=tr.tensor([2,2])
        ker=(2.0,2.0)
        kerfunc=rbf_logrbf_s_w

    if kernelname==rbf_deb_s1.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([20.0,20.0,20.0,20.0])
        configk=tr.tensor([2,2,2,2])
        #ker=(50.0,1.1,50.0,1.0,1.0)
        ker=(5.0,5.0,10.0,10.0)
        kerfunc=rbf_deb_s1

    elif kernelname==rbf_deb_s_w.__name__: 
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([11.0,6.0])
        configk=tr.tensor([2,2])
        ker=(5.0,5.1)
        kerfunc=rbf_deb_s_w


    elif kernelname==rbf_logrbf_no_s.__name__:
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([10.0,10.0])
        configk=tr.tensor([2,2])
        ker=(0.1,0.1)
        kerfunc=rbf_logrbf_no_s

    elif kernelname==KrbfMat.__name__: 
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([11.0,2.0])
        configk=tr.tensor([2,2])
        ker=(3.1,1.1)
        kerfunc=KrbfMat

    elif kernelname==Krbflog.__name__:
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([10.0,10.0])
        configk=tr.tensor([2,2])
        ker=(4.0,1.0)
        kerfunc=Krbflog

    elif kernelname==Krbflog_no_s.__name__:
        meank=tr.tensor([0.0])
        sigmak=tr.tensor([10.0])
        configk=tr.tensor([2])
        ker=(1.1,)
        kerfunc=Krbflog_no_s
        
    elif kernelname==Krbf_no_s.__name__:
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([10.0,10.0])
        configk=tr.tensor([2,2])
        ker=(0.1,0.1)
        kerfunc=Krbf_no_s

    elif kernelname==Krbf_fast.__name__:
        meank=tr.tensor([-6.0,6.0])
        sigmak=tr.tensor([12.0,12.0])
        configk=tr.tensor([2,2])
        ker=(0.0,0.0)
        kerfunc=Krbf_fast

    elif kernelname==Kdebbio.__name__:
        meank=tr.tensor([0.0,0.0])
        sigmak=tr.tensor([10.0,10.0])
        configk=tr.tensor([2,2])
        ker=(5.0,9.1)
        kerfunc=Kdebbio

    elif kernelname==Kdebbioxa.__name__:
        meank=tr.tensor([0.0,0.0,-1.0])
        sigmak=tr.tensor([10.0,10.0,2.0])
        configk=tr.tensor([2,2,2])
        ker=(3.0,1.1,-0.6)
        kerfunc=Kdebbioxa

    elif kernelname==Kdebbioxb.__name__:
        meank=tr.tensor([0.0,0.0,0.0])
        sigmak=tr.tensor([11.0,10.0,5.0])
        configk=tr.tensor([2,2,2])
        ker=(3.0,2.1,3.0)
        kerfunc=Kdebbioxb

    elif kernelname==KSM.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([10.0,5.0,10.0,10.0,5.0,10.0])
        configk=tr.tensor([2,2,2,2,2,2])
        ker=(10.0,0.5,4.0,10.0,1.5,3.0)
        kerfunc=KSM
    elif kernelname==Kpoly.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([11.0,1.0,10.0,10.0])
        configk=tr.tensor([2,2,2,2])
        ker=(10.0,0.5,1.0,5.0)
        kerfunc=Kpoly
    elif kernelname==Kpoly2.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([10.0,10.0,10.0])
        configk=tr.tensor([2,2,2])
        ker=(9.0,1.0,5.0)
        kerfunc=Kpoly

    elif kernelname==log_jac.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([11.0,1.0,10.0,10.0,10.0,10.0,2.0])
        configk=tr.tensor([2,2,2,2,2,2,2])
        ker=(10.0,0.5,2.0,1.0,9.0,1.0,1.0)
        kerfunc=log_jac
    elif kernelname==jacobi.__name__:
        meank=tr.tensor([0.0,0.0,0.0,0.0])
        sigmak=tr.tensor([20.0,1.0,20.0,20.0])
        configk=tr.tensor([2,2,2,2])
        ker=(10.0,0.5,1.0,1.0)
        kerfunc=jacobi

    if nugget=="yes":
        meank=tr.cat((meank,tr.tensor([0.0])))
        sigmak=tr.cat((sigmak,tr.tensor([2.0])))
        configk=tr.cat((configk,tr.tensor([2.0])))
        ker=ker+(0.01,)

    #stack the spec model and kernel
    if mode=="mean":
        mean=meanf
        sigma=sigmaf
        config=configf
    elif mode=="kernel":
        mean=meank
        sigma=sigmak
        config=configk
    elif mode=="all":
        mean=tr.cat((meanf,meank))
        sigma=tr.cat((sigmaf,sigmak))
        config=tr.cat((configf,configk))
    return mean,sigma,config,mod,ker,modfunc,kerfunc,device,mode,ID