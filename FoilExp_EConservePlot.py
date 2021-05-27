#%%
import pickle


class Foilproperty:
    def __init__(self, Tt,Phi1_all, Phi2_all, Phij_end,xj_lall,xj_rall,xi_lall,xi_rall,ni_lall,vj_fl,Ej_fl,\
        Ei_fl,Ei_fr,enum):
        self.Tt = Tt
        
        self.Phi1_all = Phi1_all
        self.Phi2_all=Phi2_all
        self.Phij_end = Phij_end
        self.xj_lall=xj_lall
        self.xj_rall=xj_rall
        self.xi_lall=xi_lall
        self.xi_rall=xi_rall
        self.ni_lall=ni_lall
        self.vj_fl=vj_fl
        self.Ej_fl=Ej_fl
       
        self.Ei_fl=Ei_fl
        self.Ei_fr=Ei_fr
        self.enum=enum
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            # pickle dumps the data in binary format
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        """
        This loads a data from a file
        
        usage:
        data = Data.open("some_file.pickle")
        """
        with open(filename, 'rb') as f:
             return pickle.load(f)








import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spc


Dataset1=pickle.load(open("Foilprop_shortfoilEConserve.pickle","rb"))


#%% Load Data
