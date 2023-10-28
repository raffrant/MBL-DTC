from scipy.fft import fft 					#import necessary packages
from quspin.operators import hamiltonian, exp_op
from quspin.basis import spin_basis_general
import matplotlib.pyplot as plt
import numpy as np

def drive1(t,omega):					#Time-dependent driving for the DTC
	return np.cos(omega*t)


nT=10**4					# number of Floquet periods
N=6						#number of spins in the quantum system (6-qubit spin chain)
basis = spin_basis_general(N)			# define the Pauli basis for the problem


szall=[]
for i in range(N):                                                                         #Define magnetization for each qubit s^{z}_{1,2,3,4,5,6}
 szall.append(hamiltonian([['z',[[1.0,i]]]],[],basis=basis,check_symm=False).toarray())

prob=np.zeros(nT+1) 					# initialize return probability array and magnetization array to plot the outcomes
szexp=np.zeros(nT+1)


for k in range(100):                                      #Average through k disorder realizations
    j=2.5*2*np.pi 
    jall=[[j/4,i,i+1] for i in range(N-1)]
    
    g=600*2*np.pi
    db=np.random.normal(loc=0,scale=9*2*np.pi,size=N)
    bo=5000*2*np.pi
    b=[[0.5*(bo+g*(i)+db[i]),i] for i in range(N)]
    print(b)
    staticab=[['xx',jall],['yy',jall],['zz',jall],['z',b]]
    dynamic=[]
    ham = hamiltonian(staticab,dynamic,basis=basis,dtype=np.complex128,check_symm=False)         #Define static many-body Hamiltonian through Quspin package
    
    h=0.1
    
    epsilon_satellite=0.1   #e=0.0                                              #Define time-dependent error driving for the realization of DTC through Quspin package
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
    flipv = [ [-1.0j*(np.pi/2-epsilon_satellite), i] for i in range(N) ] # satellite error
    static1 = [['x',flipv],]
    dynamic1 = []
        
    spinflip_ham1 = hamiltonian(static1,dynamic1,basis=basis,dtype=np.complex128,check_herm=False,check_symm=False)
    
    spinflip_op1 = exp_op(spinflip_ham1,a=1.0).get_mat(dense=True)            #Define the propagators of static many-body Hamiltonian and time-dependent driving
    T=0.1
    Ut = exp_op(ham,a=-1.0j*T).get_mat(dense=True)
    i7=basis.index("010101")                                                #Define the initial state as a Neel state
    
    psi0=np.zeros(2**N)
    psi0[i7]=1
    
    szexp[0]+=np.vdot(psi0,np.dot(np.array(szall[3]),psi0))                 #Calculate magnetizations and average through disorder realizations
    prob[0]+=1
    print(szexp[0])
    for i in range(nT):
        if i==0:                                                                       #Propagate the system for nT Floquet cycles
            psit=Ut.dot(psi0)
            psit=ham2.evolve(psit,(i)*T-h*T,(i)*T)
        else:
            psit=Ut.dot(psit)
            psit=ham2.evolve(psit,(i)*T-h*T,(i)*T)
            print(i)
        prob[i+1]+=(abs(np.vdot(psi0,psit))**2)    
        szexp[i+1]+=((-1)**(i+1))*(np.vdot(psit,np.dot(np.array(szall[3]),psit)))
        print(prob[i+1]/(k+1))    


plt.plot(range(nT+1),szexp/k)                         #Plot results
plt.show()
