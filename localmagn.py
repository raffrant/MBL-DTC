from scipy.fft import fft
from quspin.operators import hamiltonian, exp_op
from quspin.basis import spin_basis_general
import matplotlib.pyplot as plt
import numpy as np
import heapq
from matplotlib.collections import LineCollection
import matplotlib.cbook as cbook
#from matplotlib import cm
from numba import njit
import cmocean

def drive1(t,omega):
	return np.cos(omega*t)


nT=10**4
N=6
basis = spin_basis_general(N)
szall=[]
for i in range(N):
 szall.append(hamiltonian([['z',[[1.0,i]]]],[],basis=basis,check_symm=False).toarray())
prob=np.zeros(nT+1) 
szexp=np.zeros(nT+1)
for k in range(3):
    j=2.5*2*np.pi 
    jall=[[j/4,i,i+1] for i in range(N-1)]
    
    g=600*2*np.pi
    db=np.random.normal(loc=0,scale=9*2*np.pi,size=N)
    bo=5000*2*np.pi
    b=[[0.5*(bo+g*(i)+db[i]),i] for i in range(N)]
    print(b)
    staticab=[['xx',jall],['yy',jall],['zz',jall],['z',b]]
    dynamic=[]
    ham = hamiltonian(staticab,dynamic,basis=basis,dtype=np.complex128,check_symm=False) 
    
    h=0.1
    epsilon_satellite=0.1   #e=0.0
    omega=[bo+g*(i) for i in range(N)]
    drive_args=[[omega[i]] for i in range(N)]
    #print(drive_args[0])
    flipv = [[[200*(np.pi/2-epsilon_satellite), i]] for i in range(N)]#=# for i in range(N)] # satellite error
    #flipv = [ [10**4*N, 0]] +flipv
    #flipall=[[[np.pi,i]] for i in range(N)]
    dynamica=[['x',flipv[0],drive1,drive_args[0]]]# for i in range(N)]
    dynamica+=[['x',flipv[1],drive1,drive_args[1]]]
    dynamica+=[['x',flipv[2],drive1,drive_args[2]]]
    dynamica+=[['x',flipv[3],drive1,drive_args[3]]]
    dynamica+=[['x',flipv[4],drive1,drive_args[4]]]
    dynamica+=[['x',flipv[5],drive1,drive_args[5]]]
    #print(dynamica)
    ham2=hamiltonian(staticab,dynamica,basis=basis) 
#0.1#*np.random.random_sample((N-1,))
    flipv = [ [-1.0j*(np.pi/2-epsilon_satellite), i] for i in range(N) ] # satellite error
    static1 = [['x',flipv],]
    dynamic1 = []
        
    spinflip_ham1 = hamiltonian(static1,dynamic1,basis=basis,dtype=np.complex128,check_herm=False,check_symm=False)
    spinflip_op1 = exp_op(spinflip_ham1,a=1.0).get_mat(dense=True)
    T=0.1
    Ut = exp_op(ham,a=-1.0j*T).get_mat(dense=True)
    i7=basis.index("010101")
    psi0=np.zeros(2**N)
    psi0[i7]=1
    szexp[0]+=np.vdot(psi0,np.dot(np.array(szall[3]),psi0))
    prob[0]+=1
    print(szexp[0])
    for i in range(nT):
        if i==0:
            psit=Ut.dot(psi0)
            #psit=spinflip_op1.dot(psit)#ham2.evolve(psi0,(i+1)*T-h*T,(i+1)*T)
            psit=ham2.evolve(psit,(i)*T-h*T,(i)*T)
        else:
            psit=Ut.dot(psit)
            #psit=spinflip_op1.dot(psit)#ham2.evolve(psit,(i+1)*T-h*T,(i+1)*T)
            psit=ham2.evolve(psit,(i)*T-h*T,(i)*T)
            print(i)
        prob[i+1]+=(abs(np.vdot(psi0,psit))**2)    
        szexp[i+1]+=((-1)**(i+1))*(np.vdot(psit,np.dot(np.array(szall[3]),psit)))
        print(prob[i+1]/(k+1))    
print(szexp/3)

plt.plot(range(nT+1),szexp/3)
plt.show()
