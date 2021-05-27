#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
import pickle

n_i0=1
n_e0=1
Length=8000
Ifr=8001
Rmn=3600
Ifr_pos=10
Te0=1

xj_1=np.linspace(0,Ifr_pos,Ifr)
xj_2=np.linspace(Ifr_pos,100,Rmn+1)
xi_1=np.empty(Ifr-1)
xi_2=np.empty(Rmn)
for i in range(Ifr-1):
    xi_1[i]=0.5*(xj_1[i]+xj_1[i+1])
for i in range(Rmn):
    xi_2[i]=0.5*(xj_2[i]+xj_2[i+1])
xj=np.append(xj_1,xj_2[1:])
xi=np.append(xi_1,xi_2)

#initial ion and potential distribution
ni0_i=np.heaviside(-(xi-Ifr_pos),1)
ni0_j=np.heaviside(-(xj-Ifr_pos),1)
Phi0_1=np.zeros(Ifr-1)+1
Phi0_2=np.zeros(Rmn)+1
Phi0j_1=np.zeros(Ifr)+1
Phi0j_2=np.zeros(Rmn+1)+1

#Matrix EQN on leftside of ion front
def Matrix1(xi_l,xi_r,Phi1,phi2,T):
    
    x_temp=np.append(-xi_l[0],xi_l)
    x=np.append(x_temp,xi_r[0])
    Phi=np.append(Phi1,phi2)
    A=np.zeros((Ifr+1,Ifr+1))

    A[0][0] = -1/(x[1]-x[0])
    A[0][1] = 1/(x[1]-x[0])

     #last row
    f1 = 1/(4*np.sqrt(2*T))*((np.exp(Phi[-1]/T))**(1/2)+(np.exp(Phi[-2]/T))**(1/2))
    g1 = (np.sqrt(T/2))*((np.exp(Phi[-1]/T))**(1/2)+(np.exp(Phi[-2]/T))**(1/2))
    A[-1][-2] = -(1-f1*(x[-1]-x[-2]))
    A[-1][-1] = 1+f1*(x[-1]-x[-2])

    for i in range(1,Ifr):
        A[i][i-1] = 2/((x[i+1]-x[i-1])*(x[i]-x[i-1]))
        A[i][i] = -(2/((x[i+1]-x[i])*(x[i]-x[i-1])))-(np.exp(Phi[i-1]/T)/T)
        A[i][i+1] = 2/((x[i+1]-x[i-1])*(x[i+1]-x[i]))
        
    return(A)

def VecB1(xi_l,xi_r,Phi1,phi2, ni,T):
    B=np.zeros(Ifr+1)
    B[0]=0
    
    Phi=np.append(Phi1,phi2)
    for i in range(1,Ifr):
        B[i]=np.exp(Phi1[i-1]/T)*(1-(Phi1[i-1]/T))-ni[i-1]
    
     #boundary conditions
    f1 = 1/(4*np.sqrt(2*T))*((np.exp(Phi[-1]/T))**(1/2)+(np.exp(Phi[-2]/T))**(1/2))
    g1 =(np.sqrt(T/2))*((np.exp(Phi[-1]/T))**(1/2)+(np.exp(Phi[-2]/T))**(1/2))
    

    B[-1] = (f1*(Phi[-1]+Phi[-2])-g1)*(xi_r[0]-xi_l[-1])
    return(B)

def potential1(xi_l,xi_r,Phi1,phi2, ni,T):
    Phi1 = np.linalg.solve(Matrix1(xi_l,xi_r,Phi1,phi2,T),VecB1(xi_l,xi_r,Phi1,phi2, ni,T))
    return(Phi1)

def PrecisePhi1(xi_l,xi_r,Phi1,phi2, ni,T):
    Phi_new=np.zeros((9,Ifr+1))
    Phi_new[0] = np.linalg.solve(Matrix1(xi_l,xi_r,Phi1,phi2,T),VecB1(xi_l,xi_r,Phi1,phi2, ni,T))

    for i in range(8):
        Phi_new[i+1] = potential1(xi_l,xi_r,Phi_new[i][1:-1],Phi_new[i][-1], ni,T)
    return(Phi_new[-1])

Phi_1=PrecisePhi1(xi_1,xi_2,Phi0_1,Phi0_2[0],ni0_i,Te0)[1:]



def PrecisePhi2(Phi1,Phi2,xi,xj,phi_bdr,T):
    Phi=np.zeros(Rmn+1)
    x=np.append(xi,xj[-1])
    Phi[0]=Phi1[-1]
    for i in range(1,Rmn+1):
        Phi[i]=Phi[0]-2*T*np.log(np.exp(Phi[0]/(2*T))*(x[i]-x[0])/np.sqrt(2*T)+1)   
    return(Phi[:-1],Phi[-1])

Phi_2,phi_end=PrecisePhi2(Phi_1,Phi0_2,xi_2,xj_2,Phi0j_2[-1],Te0)
E_end=np.sqrt(2*Te0)*np.exp(phi_end/(2*Te0))

Phi=np.append(Phi_1,Phi_2[1:])
plt.plot(xi,Phi)
plt.show()

def field_fl(x_i1,x_i2, Phi1): #field on the left side of ion front
    N = Ifr
    x_i=np.append(x_i1,x_i2[0])
    E = np.zeros(N-1)
    E[0]=-0.5*(Phi1[1]-Phi1[0])/(x_i1[1]-x_i1[0])
    for i in range(1,N-1):
        E[i] = -0.5*((Phi1[i]-Phi1[i+1])/(x_i[i]-x_i[i+1])+(Phi1[i]-Phi1[i-1])/(x_i[i]-x_i[i-1]))  
    return(E)
E_fl=field_fl(xi_1,xi_2,Phi_1)

def field_fr(x_i1,x_i2, Phi1,Phi2,T):
    N = Rmn
    
    E=np.sqrt(2*T)*np.exp(Phi2/(2*T))
    return(E)

E_fr=field_fr(xi_1,xi_2,Phi_1,Phi_2,Te0)

E=np.append(E_fl,E_fr)
plt.plot(xi,E)
plt.show()


def fieldj_fl(xj,ni,Phi,T):
    Ej=np.zeros(Ifr)
    for i in range(Ifr-1):
        Ej[i+1]=Ej[i]-(np.exp(Phi[i]/T)-ni[i])*(xj[i+1]-xj[i])
    return(Ej)



#%%
#Having done with the initial condition
#We now deal with time evolution of the system
def time1(Phi1,Phi2, xj_l,xj_r, xi_l,xi_r, E1,E2,n_ion0l,n_ion0r,T,E_last,Phi_last):
    #time span for each time steps
    dt = 0.025
    dT=0 #Temperature change
    N=Ifr
    M=Rmn

    Tt=np.zeros(5000) #Temperature with repect to time
    Tt[0]=T #1MeV
    Tt[1]=T

    enum=np.zeros((3000,2)) #Number of electrons, to monitor conservation of electrons

    #all the values are stored in matrices, where the values at time j*dt are in the jth column
    #First, initialize these arrays
    Phi1_all = np.zeros((N,5000)) #Array describing potential left to the ion-front on i-grid system
    Phi1_all[0][0]=Phi1[0]          
    Phi1_all[:,0]=Phi1   
    Phij_end=np.zeros(5000)       
    
    Phi2_all=np.zeros((M,5000))    #Potential on grid points at the right side of ion-front on i-grid system
    Phi2_all[:,0]=Phi2
    Phij_end[0]=Phi_last 

    xj_lall = np.zeros((N,5000))   #j-grids left to the ion-front
    xj_lall[0][0]=-xj_l[0]
    xj_lall[:,0] = xj_l

    xj_rall=np.zeros((M+1,5000))   #j-grids right to the ion-front
    xj_rall[:,0]=xj_r
    
    xi_lall=np.zeros((N-1,5000))   #i-grids left to the ion-front
    xi_lall[:,0]=xi_l
    xi_rall=np.zeros((M,5000))
    xi_rall[:,0]=xi_r

    ni_lall=np.zeros((N-1,5000))
    ni_lall[:,0]=n_ion0l
    

    vj_fl = np.zeros((N,3000))
   
    vi_fl=np.zeros((N-1,3000))
    

    Ej_fl=np.zeros((N,5000)) 
    
    Ei_fl=np.zeros((N-1,5000))    
    Ei_fr=np.zeros((Rmn,5000))

    Ei_fl[:,0]=E1
    Ei_fr[:,0]=E2

    Ej_fl[:,0]=fieldj_fl(xj_l,n_ion0l,Phi1,Te0)
    
    
    

   
    #Time evolution with the leapfrog method
    for t in range(4):

        #Velocity
        for j in range(N):
            vj_fl[j][t+1]=vj_fl[j][t]+Ej_fl[j][t]*dt
        
        for i in range(N-1):
            vi_fl[i][t+1]=0.5*(vj_fl[i][t+1]+vj_fl[i+1][t+1])
       
        #j-grid position
        for j in range(N):
            xj_lall[j][t+1]=xj_lall[j][t]+vj_fl[j][t]*dt+0.5*Ej_fl[j][t]*dt*dt
        
        xj_rall[0][t+1]=xj_lall[-1][t+1]
        for j in range(1,M+1):
            xj_rall[j][t+1]=xj_rall[j][t]+xj_lall[-1][t+1]-xj_lall[-1][t]	
        
        #i-grid position
        for i in range(N-1):
            xi_lall[i][t+1]=0.5*(xj_lall[i][t+1]+xj_lall[i+1][t+1])
        
        for i in range(M):
            xi_rall[i][t+1]=0.5*(xj_rall[i][t+1]+xj_rall[i+1][t+1])
        
        #ion density, on i-grid of course
        
        for i in range(N-1):
            ni_lall[i][t+1]=ni_lall[i][t]*((xj_lall[i+1][t]-xj_lall[i][t])/(xj_lall[i+1][t+1]-xj_lall[i][t+1]))
       
        tail=0
        for i in range(N-1):
            enum[t][0]=enum[t][0]+np.exp(Phi1_all[i][t]/Tt[t])*(xj_lall[i+1][t]-xj_lall[i][t])
        
        enum[t][0]=enum[t][0]+Ej_fl[-1][t] #Ej_fl[-1][t] happends to be equal to Ne after ion front quantitively
           
        if t>=1:
            
            energy1=2*Tt[t-1]*Ej_fl[-1][t-1]
            energy2=2*Tt[t]*Ej_fl[-1][t]
           
            xj_a=np.append(xj_lall[:,t-1],xj_rall[1:,t-1])
            xj_b=np.append(xj_lall[:,t],xj_rall[1:,t])
            Ej_a=Ej_fl[:,t-1]
            Ej_b=Ej_fl[:,t]
            vj_a=vj_fl[:,t-1]
            vj_b=vj_fl[:,t]
            na=ni_lall[:,t-1]
            nb=ni_lall[:,t]
            for i in range(N-1):
                energy1=energy1+((xj_a[i+1]-xj_a[i])/3)*(Ej_a[i+1]**2+(Ej_a[i]**2)+Ej_a[i+1]*Ej_a[i])
                energy2=energy2+((xj_b[i+1]-xj_b[i])/3)*(Ej_b[i+1]**2+(Ej_b[i]**2)+Ej_b[i+1]*Ej_b[i])
            
            for i in range(N-1):
                energy1=energy1+(na[i]/3)*(xj_a[i+1]-xj_a[i])*(vj_a[i+1]**2+(vj_a[i])**2+vj_a[i+1]*vj_a[i])
                energy2=energy2+(nb[i]/3)*(xj_b[i+1]-xj_b[i])*(vj_b[i+1]**2+(vj_b[i])**2+vj_b[i+1]*vj_b[i])

           
           
            dT=(1/xj_lall[-1,0])*(energy1-energy2)
           
           
            Tt[t+1]=Tt[t]+dT
            print("dT=",dT)
            enum[t][1]=enum[t][0]-enum[t-1][0]
        
        if Tt[t+1]<0:
            print("Error!T<0")
            break
        
        
        Phi1_all[:,t+1]=PrecisePhi1(xi_lall[:,t+1],xi_rall[:,t+1],Phi1_all[:-1,t],Phi2_all[0,t],ni_lall[:,t+1],Tt[t+1])[1:]
        Phi2_all[:,t+1],Phij_end[t+1]=PrecisePhi2(Phi1_all[:,t+1],Phi2_all[:,t],xi_rall[:,t+1],xj_rall[:,t+1],Phij_end[t],Tt[t+1])
        
        Ei_fl[:,t+1]=field_fl(xi_lall[:,t+1],xi_rall[:,t+1],Phi1_all[:,t+1])
        Ei_fr[:,t+1]=field_fr(xi_lall[:,t+1],xi_rall[:,t+1],Phi1_all[:,t+1],Phi2_all[:,t+1],Tt[t+1])

        

        Ej_fl[:,t+1]=fieldj_fl(xj_lall[:,t+1],ni_lall[:,t+1],Phi1_all[:,t+1],Tt[t+1])
        
        
        
        
       
       
        print("Number of Electrons:",enum[t])
        
        print("Next Temperature:", Tt[t+1])
        
        plt.plot(np.append(xi_lall[:,t+1],xi_rall[:,t+1]),np.append(Phi1_all[:-1,t+1],Phi2_all[:,t+1]))
        plt.title("Potential")
        plt.show()
        plt.plot(np.append(xi_lall[:,t+1],xi_rall[:,t+1]),np.append(Ei_fl[:,t+1],Ei_fr[:,t+1]))
        plt.title("Electric field")
        plt.show()
        plt.plot(np.append(xi_lall[:,t+1],xi_rall[:,t+1]),np.append(ni_lall[:,t+1],np.zeros(Rmn)),label='ni')
        ne=np.append(Phi1_all[:-1,t+1],Phi2_all[:,t+1])
        plt.plot(np.append(xi_lall[:,t+1],xi_rall[:,t+1]),np.exp(ne/Tt[t+1]),label='ne')
        plt.title("charge distribution")
        plt.legend()
        plt.show()
        

    return(Tt,Phi1_all, Phi2_all, Phij_end,xj_lall,xj_rall,xi_lall,xi_rall,ni_lall,vj_fl,Ej_fl,\
        Ei_fl,Ei_fr,enum)

T_t,Pot1,Pot2,Pot_end,jgrid_lall,jgrid_rall,igrid_lall,igrid_rall,ion_lall,velj_fl,\
    Elcj_fl,Elci_fl,Elci_fr,ENum=time1(Phi_1,Phi_2,xj_1,xj_2,xi_1,xi_2,E_fl,E_fr,ni0_i[:Ifr-1],ni0_i[Ifr-1:],Te0,E_end,phi_end)








	