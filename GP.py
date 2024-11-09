
# General GP V1.2?
import torch as tr 
import numpy as np
import scipy.integrate as integrate
from torch.special import gammaln
import torch.nn.functional as F
from HMC import *

# I have modified the class to include the possibility of a prior in the second level of inference
class GaussianProcess():
    def __init__(self,x_grid,V,Y,Gamma,name,device,flag="noiseless",Pd= lambda x : 2.*(1.-x) ,Ker = lambda x: tr.outer(x,x),**args):
        
        
        self.name=name
        self.device= device
        self.flag=flag#define if you consider noisy of noiseless data
        self.x_grid = tr.tensor(x_grid).to(tr.float32).to(self.device)
        self.N = x_grid.shape[0]
        self.V = tr.tensor(V).to(tr.float32).to(self.device)
        self.Y = tr.tensor(Y).to(tr.float32).to(self.device)
        #print("flag")
        self.Gamma = tr.tensor(Gamma).to(tr.float32).to(self.device) # data covariance
        self.Pd = Pd # the default model function. It must work in torch for training?
        self.Ker = Ker 
        self.pd_args = tuple([tr.tensor([a]) for a in args["Pd_args"]])
        self.ker_args = tuple([tr.tensor([a]) for a in args["Ker_args"]])
        
        ### extract the error as a hyperparameter
        if self.flag=="noisy":
            self.sig = self.ker_args[-1] 
            self.ker_args = self.ker_args[:-1]
        else:
            self.sig = 1.0
        self.input_pd_args = self.pd_args
        self.input_ker_args = self.ker_args

        self.Npd_args = len(self.pd_args)
        self.Nker_args = len(self.ker_args)
        
    def ComputePosterior(self): # computes the covariance matrix of the posterior
        K = self.Ker(self.x_grid,*self.ker_args)
        Pd = self.Pd(self.x_grid,*self.pd_args)
        
        if self.flag=="noisy":
            m=tr.ones(self.Gamma.shape[0]-2,self.Gamma.shape[1]-2)
            mask=F.pad(m,(2,0,2,0),value=0)
            Chat = self.Gamma*(1-mask) + (self.sig**2+1e-7)*mask*self.Gamma +self.V@K@self.V.T
            #Chat = self.sig**2*mask*self.Gamma +(1-mask)*self.Gamma + self.V@K@self.V.T
            #Chat= self.Gamma + (self.sig**2)*(tr.eye(self.Gamma.shape[0]))*mask + self.V@K@self.V.T
            #Chat = self.Gamma + (self.sig**2)*(tr.eye(self.Gamma.shape[0])) + self.V@K@self.V.T
        else:
            Chat = self.Gamma + self.V@K@self.V.T
        #Chat = self.sig**2*self.Gamma + self.V@K@self.V.T
        iChat = tr.linalg.inv(Chat)
        VK = self.V@K
        #print(K)
        self.CpMat = K - VK.T@iChat@VK
        self.Pm = Pd +VK.T@iChat@(self.Y-self.V@Pd)
        return self.Pm,self.CpMat
    def Uncertaintyanalysis(self,trace):
        qofx=tr.zeros_like(self.x_grid)
        Z=tr.tensor([0.0])
        Pms=[]
        for i in range(trace.shape[0]):
            pd_args = tuple(trace[i,:self.Npd_args])
            k_args = tuple(trace[i,self.Npd_args:])
            E=self.nlpEvidence(pd_args,k_args)
            if self.flag=="noisy":
                sig = k_args[-1]
                #sig=10**sig
                k_args = k_args[:-1]
            else:
                sig = 1.0
            K = self.Ker(self.x_grid,*k_args)
            Pd = self.Pd(self.x_grid,*pd_args)
            if self.flag=="noisy":
                m=tr.ones(self.Gamma.shape[0]-2,self.Gamma.shape[1]-2)
                mask=F.pad(m,(2,0,2,0),value=0)
                Chat = self.Gamma*(1-mask) + (sig**2+1e-7)*mask*self.Gamma +self.V@K@self.V.T
            else:
                Chat = self.Gamma + self.V@K@self.V.T
            iChat = tr.linalg.inv(Chat)
            VK = self.V@K
            #CpMat = K - VK.T@iChat@VK
            Pm = Pd +VK.T@iChat@(self.Y-self.V@Pd)
            Pms.append(Pm)
            qofx+=tr.exp(-E)*Pm
            Z+=tr.exp(-E)
        
        qs=qofx/Z
        #Calculate the covariance matrix
        cov = tr.zeros((self.x_grid.shape[0],self.x_grid.shape[0]))
        Z=tr.tensor([0.0])
        for i in range(trace.shape[0]):
            pd_args = tuple(trace[i,:self.Npd_args])
            k_args = tuple(trace[i,self.Npd_args:])
            E=self.nlpEvidence(pd_args,k_args)
            if self.flag=="noisy":
                sig = k_args[-1]
                #sig=10**sig
                k_args = k_args[:-1]
            else:
                sig = 1.0
            K = self.Ker(self.x_grid,*k_args)
            Pd = self.Pd(self.x_grid,*pd_args)
            if self.flag=="noisy":
                m=tr.ones(self.Gamma.shape[0]-2,self.Gamma.shape[1]-2)
                mask=F.pad(m,(2,0,2,0),value=0)
                Chat = self.Gamma*(1-mask) + (sig**2+1e-7)*mask*self.Gamma +self.V@K@self.V.T
            else:
                Chat = self.Gamma + self.V@K@self.V.T
            iChat = tr.linalg.inv(Chat)
            VK = self.V@K
            CpMat = K - VK.T@iChat@VK
            Pm = Pd +VK.T@iChat@(self.Y-self.V@Pd)
            mean=Pm-qs
            qx=mean.view(1,mean.shape[0])
            qxx=qx.view(mean.shape[0],1)
            cov+=tr.exp(-E)*(0.5*CpMat+(qx)*(qxx))
            Z+=tr.exp(-E)

        return qs,cov/Z,Pms
    
    def nucovariance(self):
        #define nu grid for the covariance matrix
        nu=tr.linspace(0,100,10000)
        #compute the covariance matrix in x
        K = self.Ker(self.x_grid,*self.ker_args)
        return K
        
    #define the pdf as a function of kernel and parameters
    def posteriorpdf(self,p_x,k_x):
        # p = [a,b,sig,w]
        pd_args = tuple(p_x)
        k_args = tuple(k_x)
        K = self.Ker(self.x_grid,k_args)
        Pd = self.Pd(self.x_grid,pd_args)
        Chat = self.Gamma + self.V@K@self.V.T
        iChat = tr.linalg.inv(Chat)
        VK = self.V@K
        #Posterior covariance
        Cov = K - VK.T@iChat@VK
        Cov = (Cov + Cov.T)/2+1e-6*tr.eye(Cov.shape[0])
        Mean = Pd +VK.T@iChat@(self.Y-self.V@Pd)
        return tr.distributions.multivariate_normal.MultivariateNormal(Mean,Cov)

    #This also defines the likelihood in the second level of inference
    def nlpEvidence(self,p_x,k_x):
        # p = [a,b,sig,w]
        pd_args = tuple(p_x)
        k_args = tuple(k_x)
        """if k_x[4]<0:
            return tr.tensor([0.0],requires_grad=True)"""
        if self.flag=="noisy":
            sig = k_args[-1]
            #sig=10**sig
            k_args = k_args[:-1]
        else:
            sig = 1.0
        K = self.Ker(self.x_grid,*k_args)
        Pd = self.Pd(self.x_grid,*pd_args)
        
        m=tr.ones(self.Gamma.shape[0]-2,self.Gamma.shape[1]-2)
        mask=F.pad(m,(2,0,2,0),value=0).to(self.device)
        Chat = self.Gamma*(1-mask) + (sig**2)*self.Gamma*mask +self.V@K@self.V.T
        #Chat = self.Gamma*(1-mask) + (sig**2)*(self.Gamma)*mask +self.V@K@self.V.T# +1e-7*tr.eye(self.Gamma.shape[0]).to(self.device)
        #m=tr.ones(self.Gamma.shape[0]-2,self.Gamma.shape[1]-2)
        #mask=F.pad(m,(2,0,2,0),value=0).to(self.device)
        #Chat = self.Gamma+sig**2*(tr.eye(self.Gamma.shape[0]))+ self.V@(K)@self.V.T

        iChat = tr.linalg.inv(Chat)
        self.Chat = Chat.to(self.device)
        self.iChat = iChat
        #print(Y,self.V,Pd)
        D = self.Y - self.V@Pd
        # no need for D.T@iChat@D D.@iChat@D does the job...
        sign,logdet = tr.linalg.slogdet(Chat)
        nlp = 0.5*(D@iChat@D + sign*logdet)
        return nlp
    
    def train(self,Nsteps=100,lr=0.4,mode="kernel",function="evidence"):
        #set initial values of the parameters
        p_x=tr.tensor(self.pd_args ,requires_grad=True)
        if self.flag=="noisy":
            k_x=tr.tensor(self.ker_args+(self.sig,),requires_grad=True)
        else:
            k_x=tr.tensor(self.ker_args,requires_grad=True)
        #sig_x=tr.tensor([self.sig],requires_grad=True)
        #X=tr.cat((p_x,k_x))
        self.mode=mode

        optim = tr.optim.Adam([p_x,k_x], lr=lr) # train everything
        if mode=="kernel" :
            optim = tr.optim.Adam([k_x], lr=lr) # train only kernel
            print("Training kernel only",self.name)
        elif mode=="mean" :
            print("Training mean only",self.name)
            optim = tr.optim.Adam([p_x], lr=lr) # train only default model
        else:
            print("Training everything",self.name)
        losses = []
        for i in range(Nsteps):
            optim.zero_grad()
            if function=="evidence":
                loss = self.nlpEvidence(p_x,k_x)
            elif function=="posterior":

                if mode == "kernel":
                    X=k_x.requires_grad_(True)
                elif mode == "mean":
                    X=p_x.requires_grad_(True)
                else:
                    X=tr.cat((p_x,k_x)).requires_grad_(True)
                loss = self.post2levelpdf(X)
                
            else:
                if mode == "kernel":
                    X=k_x.requires_grad_(True)
                elif mode == "mean":
                    X=p_x.requires_grad_(True)
                else:
                    X=tr.cat((p_x,k_x)).requires_grad_(True)

                loss = self.nlogpost2levelpdf(X)
            if loss.isnan():
                print("NaN detected")
                break
            
            loss.backward()
            optim.step()
            """delta= tr.abs(p_x.detach()-tr.tensor(self.pd_args))
            if delta.sum()<1e-3:
                print("Converged")
                break"""
        losses.append(loss.detach().item()+self.Gamma.shape[0]/ 2.0*np.log(2.0*np.pi))
        self.pd_args=tuple(p_x.detach())
        self.ker_args=tuple(k_x.detach())
        #extract the error
        if self.flag=="noisy":

            self.sig = self.ker_args[-1] 
            self.ker_args = self.ker_args[:-1]
        self.losses=losses
        return losses
    
    def prior2ndlevel(self,mode,Nsam,mean,sigma,prior_mode=tr.tensor([1,2])):
        #remember that prior mode define the type of prior for each parameter 0=gaussian, 1=lognormal, 2=expbeta

        self.mode=mode
        self.prior_mode=prior_mode
        #For the case of the jacobi kernel we have to add a contraint in the interval 
        #the diagonal depends on the kernel type
        cov = tr.diag(sigma[:])


        self.prior_dist = []
        if self.flag=="noisy":
            self.Nparams = self.Npd_args+self.Nker_args+1
        else:
            self.Nparams = self.Npd_args+self.Nker_args
        for i in range(self.Nparams):
            if prior_mode[i]==0:
                self.prior_dist.append(multiNormal(mean[i].unsqueeze(0), cov[i,i].unsqueeze(0),self.device))
            elif prior_mode[i]==1:
                #if we are working with a model parameter shift the mean
                if i<self.Npd_args:
                    self.prior_dist.append(lognormal(mean[i],cov[i,i],self.device,1.0))
                else:
                    self.prior_dist.append(lognormal(mean[i],cov[i,i],self.device,0.0))
            else:
                self.prior_dist.append(expbeta(-0.99,-0.99,mean[i],cov[i,i], self.device))

    def hyperparametersvalues(self,burn=100,set="sampling"):
        #Mean and standard deviation of the hyperparameters
        if set=="sampling":
            self.pd_args= tuple(tr.mean(self.trace[burn:,:2],dim=0).detach())
            self.pd_std= tuple(tr.std(self.trace[burn:,:2],dim=0).detach())
            self.ker_args= tuple(tr.mean(self.trace[burn:,2:],dim=0).detach())
            self.ker_std= tuple(tr.std(self.trace[burn:,2:],dim=0).detach())
        elif set=="original":
            self.pd_args= self.input_pd_args
            self.ker_args= self.input_ker_args

    def post2levelpdf(self,X):

        p_x=tr.tensor(self.pd_args).clone().detach()
        if self.flag=="noisy":
            k_x=tr.tensor(self.ker_args+(self.sig,)).clone().detach()
        else:
            k_x=tr.tensor(self.ker_args).clone().detach()
        #k_x=tr.tensor(self.ker_args).clone().detach()
        prior=tr.tensor([1.0]).to(self.device)


        if self.mode=="kernel":
            kx = X.to(self.device)
            likelihood = tr.exp(-self.nlpEvidence(p_x,kx))
            for j in range(self.Npd_args,self.Nparams):

                prior*= self.prior_dist[j].pdf(kx[j-self.Npd_args])



        elif self.mode=="mean":
            px = X[:self.Npd_args].to(self.device)
            likelihood = tr.exp(-self.nlpEvidence(px,k_x))
            for i in range(self.Npd_args):
                prior*= self.prior_dist[i].pdf(px[i])
            #prior= self.prior_pd.pdf(px)
            #prior= tr.exp(self.prior_pd.log_prob(px))


        else:# all hyperparameters
            px = X[:self.Npd_args].to(self.device)
            kx = X[self.Npd_args:self.Nparams].to(self.device)
            likelihood = tr.exp(-self.nlpEvidence(px,kx))
#            prior=tr.torch.tensor([1.0])
            for i in range(self.Nparams):
                prior*= self.prior_dist[i].pdf(X[i])

        return likelihood*prior
    
    def nlogpost2levelpdf(self,X):
        #print(self.post2levelpdf(X))
        return -tr.log(self.post2levelpdf(X))
    
    def gradlogpost2levelpdf(self,X):#this is calculated for noisy data yet
        px=X[:self.Npd_args]
        px=tuple(px)
        kx=X[self.Npd_args:self.Nker_args+self.Npd_args]
        kx=tuple(kx)
        ###Derivative of the kernels
        dK_ds1 = Kcom_ds1(self.x_grid,*kx).type(tr.float64)
        dK_dw1 = Kcom_dw1(self.x_grid,*kx).type(tr.float64)
        dK_ds2 = Kcom_ds2(self.x_grid,*kx).type(tr.float64)
        dK_dw2 = Kcom_dw2(self.x_grid,*kx).type(tr.float64)
        dK_ds = Kcom_ds(self.x_grid,*kx).type(tr.float64)
        
        K = self.Ker(self.x_grid,*kx)
        Pd = self.Pd(self.x_grid,*px)
        Chat = self.Gamma + self.V@K@self.V.T
        iChat = tr.linalg.inv(Chat)
        #print(Y,self.V,Pd)
        D = self.Y - self.V@Pd

        ### Derivative of the model(mean)
        DP=DPDFnormed(self.x_grid,*px)
        gradEvi=[]
        i=0
        for dP in DP:
            gradEvi.append(-(self.V@dP.type(tr.float64)@iChat@D)-self.prior_pd[i].dpdf(px[i])/self.prior_pd[i].pdf(px[i]))#+self.D@self.iChat@dP.type(tr.float64)@self.V))
            i=i+1

        dK = [dK_ds1,dK_dw1,dK_ds2,dK_dw2,dK_ds]
        self.alp= iChat@D
        for i in range(self.Nker_args):
            gradEvi.append(-0.5*(self.alp.T@self.V@dK[i].type(tr.float64)@self.V.T@self.alp)+0.5*tr.trace(iChat@self.V@dK[i].type(tr.float64)@self.V.T) -self.prior_ker[i].dpdf(kx[i])/self.prior_ker[i].pdf(kx[i]))
        



        return tr.tensor(gradEvi,dtype=tr.float32)


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

def Pd(P):
    x=tr.tensor([.2])
    a,b=P[0],P[1]
    return simplePDFnormed(x,a,b)

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


#### Kernels #####
def KrbfMat(x,s,w):
    xx=x.view(1,x.shape[0])
    yy=x.view(x.shape[0],1)
    return s*s*tr.exp(-0.5*((xx - yy)/w)**2)


"""class splitRBFker():
    def __init__(self,sp,scale=1):
        self.sp =sp
        self.scale = scale
    def KerMat(self,x,s1,w1,s2,w2):
        K2 = KrbfMat(x,s2,w2) # linear
        K1 = KrbfMat(tr.log(x),s1,w1)
        sig = tr.diag(tr.special.expit(self.scale*(x-self.sp)))
        sigC = tr.eye(x.shape[0])-sig
        ##return K1+K2
        return sigC@K2@sigC + sig@K1@sig"""

def Sig(x,scale,sp=0.1):
    ss=scale**2
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
   return (s**2)*(x*y)**a*((1-x)*(1-y))**b*(R(2*x-1,t)*R(2*y-1,t)*((1-t+R(2*x-1,t))*(1-t+R(2*y-1,t)))**a*((1+t+R(2*x-1,t))*(1+t+R(2*y-1,t)))**b)**(-1)



#### Probablity Distributions ####
###################################
class lognormal():
    def __init__(self,mu,sigma,device,shift=0):
        #print("gaussian init")
        self.device=device
        self.mu = mu.to(device)
        self.sigma = sigma.to(device)
        self.shift = shift
        #self.d = sigma.shape[0]
    def pdf(self,x):
        eps=1e-15 #to avoid log(0)
        
        #if (x<tr.ones(x.shape[0])*(eps+self.shift)).any():
        if (x<eps-self.shift).item(): 
            return tr.tensor(0.0).to(self.device)
        else:
            return (tr.exp(-(tr.log(x+self.shift) -  self.mu)**2 / (2 *  self.sigma**2)) / ((x+self.shift) *  self.sigma * tr.sqrt(tr.tensor(2 * tr.pi))))
    def dpdf(self,x):
        eps=1e-10
        dPdx= -self.pdf(x)*(-self.mu + self.sigma**2+tr.log(x+self.shift))/((x+self.shift)*self.sigma**2)
        if (x<eps-self.shift).item(): 
            return tr.tensor([0.0]).to(self.device)
        else:
            return dPdx


#multivariate gaussian with torch 
class multiNormal:
    def __init__(self,mu,sigma,device):
        #print("gaussian init")
        self.device=device
        self.mu = mu.to(device)
        self.sigma = sigma.to(device)
        self.d = sigma.shape[0]
        if self.d==1:
            self.sigma = self.sigma.unsqueeze(0)
    def pdf(self,x):
        if x.shape==tr.Size([]):
            x=x.unsqueeze(0)
        return 1/(tr.sqrt(tr.tensor([(2*tr.pi)**self.d]).to(self.device)*tr.det(self.sigma)))*tr.exp(-0.5*(x-self.mu)@tr.linalg.inv(self.sigma)@(x-self.mu))
    def dpdf(self,x):
        if x.shape==tr.Size([]):
            x=x.unsqueeze(0)
        return -tr.linalg.inv(self.sigma)@(x-self.mu)*self.pdf(x)
    
class expbeta():
    def __init__(self,a,b,shift,scale,device):
        self.device=device
        self.a=tr.tensor(a).to(device)
        self.b=tr.tensor(b).to(device)
        self.shift=tr.tensor(shift).to(device)
        self.scale=tr.tensor(scale).to(device)
    ####LOGPDF (beta and dbeta is not shifted or scaled)
    def betadist(self,x):
        return (x)**(self.a)*(1-x)**(self.b)*tr.exp(gammaln(self.a+self.b+2) - gammaln(self.a+1) - gammaln(self.b+1))
    def dbetadist(self,x):
        return self.betadist(x)*((self.a)/(x) - (self.b)/(1-x))
    
    def pdf(self,x):
        if x<self.shift or x>self.shift+self.scale:
            return tr.tensor([0.0]).to(self.device)#.requires_grad_()
        return tr.exp(-self.betadist((x-self.shift)/self.scale))/self.scale
    def dpdf(self,x):
        if x<self.shift or x>self.shift+self.scale:
            return tr.tensor([0.0]).to(self.device)#.requires_grad_()
        return -self.dbetadist((x-self.shift)/self.scale)*self.pdf(x)/self.scale**2
    def nlogpdf(self,x):
        return -tr.log(self.pdf(x))
    def gradnlogpdf(self,x):
        if self.dpdf(x).item()==0:
            return tr.tensor([0.0]).to(self.device)
        else:
            return -self.dpdf(x)/self.pdf(x)
    

    def sample(self,N):
        sampler=HMC_sampler(self.nlogpdf,device,tr.tensor([0.4]),grad=self.gradnlogpdf)
        sampler.q0= (self.shift + 0.5*self.scale).unsqueeze(0).requires_grad_()
        qs,ps,H= sampler.sample(sampler.q0,N,1/20,20)
        return qs
