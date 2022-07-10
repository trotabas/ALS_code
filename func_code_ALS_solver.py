#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 18:47:42 2022

@author: trotabas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import newton
from scipy.optimize import brentq
from scipy import special
from decimal import *
#from matplotlib.ticker import Scalarformatter
#import itertools
from itertools import chain
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
from scipy.sparse import linalg 
import scipy.sparse as spsp
from scipy.integrate import quad
import time  
from scipy.interpolate import interp2d
from math import floor, log10

from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm
import pandas as pd


def ALS_solver_FDscheme_Umesh(ne,P,Te,Ti,Tn,B,Xi):
    InputFolder = "input.dat"
    dict_input = {}
    list_electrode_bias = []
    list_electrode_configuration = []
    list_electrode_radius = []
    for i in open(InputFolder).readlines():
        x = i.strip().split() 
        if x[0][0] != '/':
            if x[0][0] == '0':
                nb_electrode = int(np.float64(x[2]))
            elif x[0][0] == '1': 
                list_electrode_bias.extend( [(np.float64(x[k+2])) for k in np.arange(nb_electrode) ])
            elif x[0][0] == '2': 
                list_electrode_configuration.extend( [x[k+2] for k in np.arange(nb_electrode) ])
            elif x[0][0] == '3': 
                list_electrode_radius.extend( [(np.float64(x[k+2])) for k in np.arange(nb_electrode) ]) 
            elif x[0][0] == '4':
                BC_electrode_ground = x[2] 
            else:
                dict_input[x[0]] = np.float64(x[1])
    #==============================================================================
    #---------------------:PHYSICAL-CONSTANT---------------------------------------
    #==============================================================================
    Virtual_cathod_regime = int(dict_input.get('Virtual_cathod_regime'))
    #==============================================================================
    #---------------------:PHYSICAL-CONSTANT---------------------------------------
    #==============================================================================
    NA         = dict_input.get('NA')           # Avogadro  Number      [mol^-1] 
    kB         = dict_input.get('kB')           # Boltzmann Constant
    CoulombLog = dict_input.get('CoulombLog')   # Coulomnb  Logarithm
    eps0       = dict_input.get('eps0')         # Coulomnb  Logarithm
    sig0       = dict_input.get('sig0')         # Coulomnb  Logarithm
    e          = dict_input.get('e')            # Eletric   Charge      [C]
    me         = dict_input.get('me')           # Electron  Mass        [kg]
    #==============================================================================
    #---------------------:PLASMA-PARAMETERS---------------------------------------
    #==============================================================================
    mi_amu     = dict_input.get('mi_amu')       # Argon atom mass      [amu] 
    mi         = mi_amu*1.6605e-27              # Argon atom mass      [kg] 
    eta        = me/mi
    nN         = (1/(kB*Tn))*P                          # Neutral   Density     [m^-3]
    #==============================================================================
    #---------------------:EMISSIVE-ELECTRODE--------------------------------------
    #==============================================================================
    Tw    = dict_input.get('Tw')                # Heating    Temperature [K]
    W     = dict_input.get('W')                 # Work       Function    [J]                                 
    # =============================================================================
    # ---------------------: SHEATH PROPERTIES  -----------------------------------
    # =============================================================================
    Cs = np.sqrt((e*Te)/mi)                     # Bohm       Velocity    [m s^-1]
    Lambda = np.log(np.sqrt(1/(2*np.pi*eta)))   # Sheath     Parameter
    j_is = (e*ne*Cs)                             # Ion  Sat.  currend  density  [m^-3]
    #==============================================================================
    #---------------------:GEOMETRICAL-PARAMETERS----------------------------------
    #==============================================================================
    rg = dict_input.get('rg')                   # Column     Radius      [m]
    L = dict_input.get('L')                     # Column     Length      [m]
    #==============================================================================
    #---------------------:MESHING-PARAMETERS-PART-1------------------------------------
    #============================================================================== 0.06 0.12 0.16
    Nr     = int(dict_input.get('Nr'))          # Radial           Node      
    Nz     = int(dict_input.get('Nz'))          # Axial            Node   
    K      = Nr*Nz                              # Total            Node   
    #--> 
    step_r = rg/(Nr-1)                          # Radial    step             
    step_z = (L/2)/(Nz-1)                       # Axial     step      
    #-->
    r = np.linspace(0, rg, Nr)                  # Radial    vector 
    z = np.linspace(-L/2, 0, Nz)                # Axial    vector  
    #==============================================================================
    #---------------------:MESHING-PARAMETERS-PART-2---------------------------------------
    #==============================================================================
    list_Nr       = np.zeros(nb_electrode, dtype = int)
    list_Nr_rshow = np.zeros(nb_electrode, dtype = int)
    for index in np.arange(nb_electrode):
        list_Nr[index]       = np.where(r >= list_electrode_radius[index])[0][0]
        list_Nr_rshow[index] = np.where(r >= list_electrode_radius[index])[0][0] + 1
    list_Nr = np.insert(list_Nr,0,0)
    list_Nr = np.insert(list_Nr,list_Nr.size,Nr-1)
    list_Nr_rshow = np.insert(list_Nr_rshow,0,0)
    list_Nr_rshow = np.insert(list_Nr_rshow,list_Nr_rshow.size,Nr-1)
    #-->
    Sr = np.zeros(Nr)                           # Surface vector 
    Sr[0]   = np.pi*(step_r/2)**2
    Sr[1:]  = 2*np.pi*r[1:]*step_r    
    # =============================================================================
    # ---------------------: GEOMETRIC PARAMETERS ---------------------------------
    # =============================================================================
    start_time = time.time()
    # INITIALISATION DIRECT METHOD SPARSE
    # DIMENSION ------------------------------------------------------------------
    dim_left = 1
    for index in np.arange(nb_electrode):
        dim_left += 3*(list_Nr[index+1] - list_Nr[index])
    index = nb_electrode
    if BC_electrode_ground == "D":
        dim_left += (list_Nr[index+1] - list_Nr[index])
    else:
        dim_left += 3*(list_Nr[index+1] - list_Nr[index])
    # dim_left   = 3*(Nr-1) + 1
    dim_right  = 3*(Nr-1) + 1
    dim_top    = 3*(Nz-2)
    dim_bottom = Nz-2
    dim_inside = 5*(Nz-2)*(Nr-2)
    dim_matrix_A = dim_left + dim_right + dim_top + dim_bottom + dim_inside
    row_A  = np.zeros(dim_matrix_A)
    col_A  = np.zeros(dim_matrix_A)
    data_A = np.zeros(dim_matrix_A)
    #------------------------------------------------------------------------------
    Alpha_NBC_r = (-3)*(1/(2*step_r))
    Gamma_P_r   =    4*(1/(2*step_r))
    Gamma_PP_r  = (-1)*(1/(2*step_r))
    #~~
    Alpha_LBC = (-1)*(3/(2*step_z))
    Alpha_RBC = (-1)*(3/(2*step_z))
    Beta_P_z      =    4*(1/(2*step_z))
    Beta_PP_z     = (-1)*(1/(2*step_z))
    
    # =============================================================================
    # TOP BOUNDARY CONDITION -- NEUMANN (grad Phi |r=0  = 0) 
    # =============================================================================3*26 + 3*38 + 37
    index_A = np.where(data_A == 0)[0][0]
        #-- row
    row_A[0:3*len(np.arange(1,Nz-1)):3]   = np.arange(1,Nz-1)
    row_A[1:1+3*len(np.arange(1,Nz-1)):3] = np.arange(1,Nz-1)
    row_A[2:2+3*len(np.arange(1,Nz-1)):3] = np.arange(1,Nz-1)
        #-- column
    col_A[0:3*len(np.arange(1,Nz-1)):3]     = np.arange(1,Nz-1)
    col_A[1:1+3*len(np.arange(1,Nz-1)):3] = np.arange(1,Nz-1) + Nz
    col_A[2:2+3*len(np.arange(1,Nz-1)):3] = np.arange(1,Nz-1) + 2*Nz
        #-- data    
    data_A[0:3*len(np.arange(1,Nz-1)):3] = Alpha_NBC_r
    data_A[1:1+3*len(np.arange(1,Nz-1)):3] = Gamma_P_r
    data_A[2:2+3*len(np.arange(1,Nz-1)):3] = Gamma_PP_r 
    
    # =============================================================================
    # LEFT BC -- 1 -- EMISSIVE + BIASING ELECTRODE 
    # =============================================================================
    list_index_LBC = np.zeros(nb_electrode, dtype = int)
    for index in np.arange(nb_electrode):
        index_A = np.where(data_A == 0)[0][0]
        list_index_LBC[index] = np.where(data_A == 0)[0][0] 
            #-- row
        row_A[index_A:index_A+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3]     = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)
        row_A[index_A+1:index_A+1+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)
        row_A[index_A+2:index_A+2+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)
            #-- column
        col_A[index_A:index_A+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3]     = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)
        col_A[index_A+1:index_A+1+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz) + 1
        col_A[index_A+2:index_A+2+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz) + 2
            #-- data
        data_A[index_A:index_A+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3]     = Alpha_LBC
        data_A[index_A+1:index_A+1+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = Beta_P_z  
        data_A[index_A+2:index_A+2+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = Beta_PP_z  
    
    
    # =============================================================================
    # LEFT BC -- 1 -- DIRICHLET BC 
    # ============================================================================= np.arange(N*M_e,K-N,N)
    index_A = np.where(data_A == 0)[0][0]
    index = nb_electrode
    if BC_electrode_ground == "D":
            #-- row
        row_A[index_A:index_A+len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz))]     = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)
            #-- column
        col_A[index_A:index_A+len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz))]     = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)
            #-- data
        data_A[index_A:index_A+len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz))]    = 1 
    else:
            #-- row
        row_A[index_A:index_A+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3]      = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)
        row_A[index_A+1:index_A+1+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3]  = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)
        row_A[index_A+2:index_A+2+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3]  = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)
            #-- column
        col_A[index_A:index_A+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3]      = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz) 
        col_A[index_A+1:index_A+1+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3]  = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz) + 1
        col_A[index_A+2:index_A+2+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3]  = np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz) + 2
            #-- data
        data_A[index_A:index_A+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3]     = Alpha_LBC 
        data_A[index_A+1:index_A+1+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = Beta_P_z
        data_A[index_A+2:index_A+2+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = Beta_PP_z
    
    # =============================================================================
    # RIGHT BC -- NEUMANN (grad Phi |z=0  = 0) 
    # =============================================================================
    index_A = np.where(data_A == 0)[0][0]
        #-- row
    row_A[index_A:index_A+3*len(np.arange(Nz-1,K-Nz,Nz)):3]     = np.arange(Nz-1,K-Nz,Nz)
    row_A[index_A+1:index_A+1+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = np.arange(Nz-1,K-Nz,Nz)
    row_A[index_A+2:index_A+2+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = np.arange(Nz-1,K-Nz,Nz)
        #-- column
    col_A[index_A:index_A+3*len(np.arange(Nz-1,K-Nz,Nz)):3]     = np.arange(Nz-1,K-Nz,Nz) 
    col_A[index_A+1:index_A+1+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = np.arange(Nz-1,K-Nz,Nz) - 1
    col_A[index_A+2:index_A+2+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = np.arange(Nz-1,K-Nz,Nz) - 2
        #-- data
    data_A[index_A:index_A+3*len(np.arange(Nz-1,K-Nz,Nz)):3]     = Alpha_RBC  
    data_A[index_A+1:index_A+1+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = Beta_P_z  
    data_A[index_A+2:index_A+2+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = Beta_PP_z    
    
    # =============================================================================
    # DOWN BC 
    # =============================================================================
    index_A = np.where(data_A == 0)[0][0]
    row_A[index_A:index_A+Nz] = np.arange(K-Nz,K)
    col_A[index_A:index_A+Nz] = np.arange(K-Nz,K)
    data_A[index_A:index_A+Nz] = 1
    #--
    row_BC = np.zeros(Nr-1) 
    col_BC = np.zeros(Nr-1) 
    data_BC = np.zeros(Nr-1) 
    #--
    row_BC[:] = np.arange(0,Nz*(Nr-1),Nz)
    col_BC[:] = 0   
    #--
    vect_BC_sparse = spsp.csr_matrix((data_BC,(row_BC,col_BC)),shape=(K,1))
    #vect_BC_sparse = vect_BC_sparse.todense()
    
    # =============================================================================
    # Calcul
    # =============================================================================
    index_Inside_domain_zone1  = np.where(data_A == 0)[0][0]
    
    # =============================================================================
    # ---------------------: Cyclotron frequency ----------------------------------
    # =============================================================================
    Omega_i = (e*B)/mi
    Omega_e = (e*B)/me
    
    # =============================================================================
    # ---------------------: Collision frequency ----------------------------------
    # =============================================================================
    nu_en = sig0*nN*np.sqrt((8*e*Te)/(np.pi*me))
    nu_in = sig0*nN*np.sqrt((8*e*Ti)/(np.pi*mi))
    nu_ei = (e**(5/2)*CoulombLog/(6*np.sqrt(2)*np.pi**(3/2)*eps0**2*np.sqrt(me)))*ne*Te**(-3/2)
    
    # =============================================================================
    # ---------------------: MOBILITY and DIFFUSION -------------------------------
    # =============================================================================
    nu_ei = 0
    mu_e_para = e/(me*(nu_ei+nu_en))
    D_e_para  = Te*mu_e_para
    #-->
    mu_e_perp = (me/(e*B**2))*((nu_ei+nu_en)/(1 + (nu_ei+nu_en)**2/Omega_e**2)) 
    D_e_perp  = Te*mu_e_perp
    #-->
    mu_i_para = e/(me*(nu_in)) 
    D_i_para  = Te*mu_i_para
    #-->
    mu_i_perp = (mi/(e*B**2))*((nu_in)/(1 + (nu_in)**2/Omega_i**2))
    D_i_perp  = Te*mu_i_perp

    # =============================================================================
    # ---------------------: CONDUCTIVITY -----------------------------------------
    # =============================================================================
    sigma_perp0 = e*ne*(mu_i_perp + mu_e_perp)
    sigma_para0 = e*ne*(mu_i_para + mu_e_para)
    mu          = (sigma_perp0/sigma_para0)
    tau         = (L/rg)*np.sqrt(mu)
    
    # =============================================================================
    # ---------------------: INSIDE THE DOMAIN - ZONE 1 :  EMISSIVE + BIASED ELECTRODE 
    # =============================================================================
    vect_fiveScheme_zoneR1 = np.zeros((Nr-2)*(Nz-2)) # np.zeros((Nz*Nr)-2*Nr-2*(Nz-2))
    index_vect_fiveScheme_zoneR1 = 0
    for i in np.arange(Nz+1,Nz*(Nr-1)-1):
        if i % Nz != 0 and (i+1) % Nz != 0 :
            vect_fiveScheme_zoneR1[index_vect_fiveScheme_zoneR1] = i
            index_vect_fiveScheme_zoneR1 += 1
    #-- coef DIRECT
    Alpha_DIRECT_zoneR1    = -2*( 1/step_r**2 + 1/(mu*step_z**2) )
    Beta_DIRECT_zoneR1     = 1/(mu*step_z**2) 
    Gamma_P_DIRECT_zoneR1  = (1/step_r**2 + (1/(2*r[1:Nr-1]*step_r)))
    Gamma_M_DIRECT_zoneR1  = (1/step_r**2 - (1/(2*r[1:Nr-1]*step_r)))
    index_coef = 0
    Alpha_zoneR1 = np.zeros(Nz-2)
    Beta_zoneR1  = np.zeros(Nz-2)
    Gamma_P_zoneR1 = np.zeros(len(Gamma_P_DIRECT_zoneR1)*(Nz-2))
    Gamma_M_zoneR1 = np.zeros(len(Gamma_M_DIRECT_zoneR1)*(Nz-2))
    for i in range(len(Gamma_P_DIRECT_zoneR1)):
        for Nzz in np.arange(Nz-2):
    #            Alpha_zoneR1[index_coef] = Alpha_DIRECT_zoneR1[i]
    #            Beta_zoneR1[index_coef] = Beta_DIRECT_zoneR1[i]
            Gamma_P_zoneR1[index_coef] = Gamma_P_DIRECT_zoneR1[i]
            Gamma_M_zoneR1[index_coef] = Gamma_M_DIRECT_zoneR1[i]
            index_coef += 1
    
    # =============================================================================
    # ---------------------: INSIDE THE DOMAIN  -----------------------------------
    # =============================================================================
    # row
    row_A[index_Inside_domain_zone1:index_Inside_domain_zone1+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
    row_A[index_Inside_domain_zone1+1:index_Inside_domain_zone1+1+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
    row_A[index_Inside_domain_zone1+2:index_Inside_domain_zone1+2+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
    row_A[index_Inside_domain_zone1+3:index_Inside_domain_zone1+3+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
    row_A[index_Inside_domain_zone1+4:index_Inside_domain_zone1+4+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
    #-- column
    col_A[index_Inside_domain_zone1:index_Inside_domain_zone1+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
    col_A[index_Inside_domain_zone1+1:index_Inside_domain_zone1+1+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 + 1
    col_A[index_Inside_domain_zone1+2:index_Inside_domain_zone1+2+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 + Nz
    col_A[index_Inside_domain_zone1+3:index_Inside_domain_zone1+3+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 - 1
    col_A[index_Inside_domain_zone1+4:index_Inside_domain_zone1+4+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 - Nz
    #-- data
    data_A[index_Inside_domain_zone1:index_Inside_domain_zone1+5*len(vect_fiveScheme_zoneR1):5]     = Alpha_DIRECT_zoneR1
    data_A[index_Inside_domain_zone1+1:index_Inside_domain_zone1+1+5*len(vect_fiveScheme_zoneR1):5] = Beta_DIRECT_zoneR1
    data_A[index_Inside_domain_zone1+2:index_Inside_domain_zone1+2+5*len(vect_fiveScheme_zoneR1):5] = Gamma_P_zoneR1  
    data_A[index_Inside_domain_zone1+3:index_Inside_domain_zone1+3+5*len(vect_fiveScheme_zoneR1):5] = Beta_DIRECT_zoneR1
    data_A[index_Inside_domain_zone1+4:index_Inside_domain_zone1+4+5*len(vect_fiveScheme_zoneR1):5] = Gamma_M_zoneR1  
    
    # =============================================================================
    # ---------------------: SOURCE TERM ------------------------------------------
    # =============================================================================
    if Virtual_cathod_regime:
        # Mach = 1
        # Xi_crit = lambda V,x : (1/(np.sqrt(2*(me/mi)*((x-V)/Te))))*( np.exp(-((x-V)/Te)) - 1 + Mach**2*(np.sqrt(1+(2/Mach**2)*((x-V)/Te))-1))
        Xi_crit  = lambda V,x : (1/(np.sqrt(2*(me/mi)*((x-V)/Te))))*( np.exp(-((x-V)/Te)) - 2 + np.sqrt(1+2*((x-V)/Te)))
        dXi_crit = lambda V,x : ( (-1/(2*np.sqrt(2)*((x-V)/Te)**(3/2)))*( np.exp(-((x-V)/Te)) - 2 + np.sqrt(1+2*((x-V)/Te)))
                                 + (1/(np.sqrt(2*((x-V)/Te))))*( (1/(np.sqrt(1 + 2*((x-V)/Te)))) - np.exp(-((x-V)/Te)) )    )
        #-->
        Sj_func_VC  = lambda V,x :  (j_is/sigma_para0)*(1 + Xi_crit(V,x) - np.exp(Lambda + ((V-x)/Te)) )    #
        dSj_func_VC = lambda V,x :  (j_is/(Te*sigma_para0))*( np.exp(Lambda + ((V-x)/Te)) + dXi_crit(V,x))  # 0 #

    Sj_func  = lambda V,x :  (j_is/sigma_para0)*(1 + Xi - np.exp(Lambda + ((V-x)/Te)) ) #
    dSj_func = lambda V,x :  (j_is/(Te*sigma_para0))*np.exp(Lambda + ((V-x)/Te)) # 0 #
    #--
    phi_sh_init = np.ones(Nr)
    for index in np.arange(nb_electrode):
        node = int(list_Nr[index+1]-list_Nr[index])
        phi_sh_init[list_Nr[index]:list_Nr[index+1]]    = (list_electrode_bias[index] + Lambda*Te + 100)*np.ones(node)
    #---------------------------------------------------------- DIRECT METHOD
    err_L2 = 1.0
    err_L8 = 1.0
    I_null = 1.0
    err_SjVe = 1.0
    iteration = 0
    Eps_L2 = 1e-7
    Eps_I = 1e-7
    #--
    list_I_null = np.ones(list_electrode_configuration.count("f"))
    list_phi_f  = np.ones(list_electrode_configuration.count("f"))
    print("err_L2   phi_f   I_null   I_null_1   iteration") 
    while I_null > Eps_I or err_L2 > Eps_L2: # iteration < 1: # err_L2 > Eps: # gap_L2 > Eps: # gap_L2 > Eps: # iteration < 300: 
        #==============================================================================
        #---------------------:Fill-SPARSE-MATRIX--------------------------------------
        #==============================================================================
        for index in np.arange(nb_electrode):
            index_float = 0
            
            if list_electrode_configuration[index] == "b":
                phi_b = list_electrode_bias[index]
                Alpha_LBC = (-1)*((3/(2*step_z)) + dSj_func(phi_b,phi_sh_init[list_Nr[index]:list_Nr[index+1]]) )
                data_A[list_index_LBC[index]:list_index_LBC[index]+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = Alpha_LBC
                
            if list_electrode_configuration[index] == "f":
                if iteration == 0:
                    phi_f = list_electrode_bias[index]
                else:
                    phi_f = list_phi_f[index_float]
                Alpha_LBC_float  = (-1)*((3/(2*step_z)) + dSj_func(phi_f,phi_sh_init[list_Nr[index]:list_Nr[index+1]] ) )
                data_A[list_index_LBC[index]:list_index_LBC[index]+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = Alpha_LBC_float
                index_float += 1
                
            if list_electrode_configuration[index] == "d":
                Alpha_LBC = (-1)*(3/(2*step_z)) 
                data_A[list_index_LBC[index]:list_index_LBC[index]+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = Alpha_LBC
        #----------------------------------------------------------------
        sparse_matrix_A = spsp.csr_matrix((data_A,(row_A,col_A)),shape=(K,K))
        # matrix_A = sparse_matrix_A.todense()
        #==============================================================================
        #---------------------:Fill-BOUNDARY-CONDITIONS-VECTOR-------------------------
        #==============================================================================
        index_float = 0
        for index in np.arange(nb_electrode):
            
            if list_electrode_configuration[index] == "b":
                phi_b = list_electrode_bias[index]
                data_BC[list_Nr[index]:list_Nr[index+1]] = Sj_func(phi_b,phi_sh_init[list_Nr[index]:list_Nr[index+1]]) - dSj_func(phi_b,phi_sh_init[list_Nr[index]:list_Nr[index+1]])*phi_sh_init[list_Nr[index]:list_Nr[index+1]] 

            if list_electrode_configuration[index] == "f":
                if iteration == 0:
                    phi_f = list_electrode_bias[index]
                else:
                    phi_f = list_phi_f[index_float]
                data_BC[list_Nr[index]:list_Nr[index+1]] = Sj_func(phi_f,phi_sh_init[list_Nr[index]:list_Nr[index+1]]) - dSj_func(phi_f,phi_sh_init[list_Nr[index]:list_Nr[index+1]])*phi_sh_init[list_Nr[index]:list_Nr[index+1]] 
                index_float += 1
            
            if list_electrode_configuration[index] == "d":
                data_BC[list_Nr[index]:list_Nr[index+1]] = 0
                
        if BC_electrode_ground == "D":
            index = nb_electrode
            data_BC[list_Nr[-2]:] = (np.log(r[list_Nr[index]:list_Nr[index+1]]/rg) / np.log(r[list_Nr[-2]-1]/rg))*phi_sh_init[list_Nr[-2]-1] 
        else:
            data_BC[list_Nr[-2]:] = 0
        #--
        vect_BC_sparse = spsp.csr_matrix((data_BC,(row_BC,col_BC)),shape=(K,1))
        #vect_BC_sparse = vect_BC_sparse.todense()
        #==============================================================================
        #---------------------:LINEAR SOLVE-------------------------
        #==============================================================================
    
        phi_FDscheme = spsp.linalg.spsolve(sparse_matrix_A, vect_BC_sparse)
        #phi_FDscheme = spsp.linalg.bicgstab(sparse_matrix_A, vect_BC_sparse)
        phi_FDscheme = phi_FDscheme.reshape(Nr,Nz)
    
    
        #------------------------- SAVE 
        phi_sh_init    = phi_FDscheme[:,0]  
        j_para_sh      = sigma_para0*(1/(2*step_z))*(-3*phi_FDscheme[:,0] + 4*phi_FDscheme[:,1] - phi_FDscheme[:,2]) 
        # I_null         = np.sum(j_para_sh[Nre:Nrf]*Sr[Nre:Nrf])
        #==============================================================================
        #---------------------:FIND THE FLOATING TENSION------------------------
        #==============================================================================
        index_float = 0
        for index in np.arange(nb_electrode):
            if list_electrode_configuration[index] == "f":
                find_phi_f  = lambda x: sum( -1*j_is*(1  - np.exp(Lambda + (x - phi_FDscheme[j,0] )/Te ) )*Sr[j] for j in np.arange(list_Nr[index],list_Nr[index+1]) )
                #----------------------------------> BRENT <-------------------------
                phi_f_update = brentq(find_phi_f,np.min(list_electrode_bias),-1*np.min(list_electrode_bias))
                list_phi_f[index_float]        = phi_f_update
                #---------------------------------->       <------------------------- 
                list_I_null[index_float] = np.sum(  j_is*(1  - np.exp(Lambda + (phi_f - phi_FDscheme[list_Nr[index]:list_Nr[index+1],0] )/Te ) )*Sr[list_Nr[index]:list_Nr[index+1]]  )   
                #---------------------------------->
                index_float += 1
            #--------------------------> CONVERGENCE CRITERIA 
        if iteration > 0:
            #----> NORME L2 CRITERIA 
            err_L2 = np.sqrt(np.nansum((phi_FDscheme[1:-1,1:-1]-phi_save[1:-1,1:-1])**2/(phi_FDscheme[1:-1,1:-1]**2)))
            if (iteration % 100) == 0:
                print(err_L2, iteration ) 
                print("--")
        if 'f' not in list_electrode_configuration:
            I_null = 1e-10
        else:
            I_null = np.max(np.abs(list_I_null))
        #------------------------- Save potential
        phi_save = phi_FDscheme.copy()
        #------------------------- ITERATION
        iteration +=1
        
    #==============================================================================
    #---------------------:FORMATION OF A VIRTUAL CATHODE------------------------
    #==============================================================================
    if Virtual_cathod_regime:
        err_L2 = 1.0
        err_L8 = 1.0
        I_null = 1.0
        err_SjVe = 1.0
        iteration = 0
        Eps_L2 = 1e-7
        Eps_I = 1e-7
        #--
        list_I_null = np.ones(list_electrode_configuration.count("f"))
        list_phi_f  = np.ones(list_electrode_configuration.count("f"))
        print("err_L2   phi_f   I_null   I_null_1   iteration") 
        while I_null > Eps_I or err_L2 > Eps_L2: # iteration < 1: # err_L2 > Eps: # gap_L2 > Eps: # gap_L2 > Eps: # iteration < 300: 
            #==============================================================================
            #---------------------:Fill-SPARSE-MATRIX--------------------------------------
            #==============================================================================
            for index in np.arange(nb_electrode):
                index_float = 0
                
                if list_electrode_configuration[index] == "b":
                    phi_b = list_electrode_bias[index]
                    Alpha_LBC = (-1)*((3/(2*step_z)) + dSj_func_VC(phi_b,phi_sh_init[list_Nr[index]:list_Nr[index+1]]) )
                    data_A[list_index_LBC[index]:list_index_LBC[index]+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = Alpha_LBC
                    
                if list_electrode_configuration[index] == "f":
                    if iteration == 0:
                        phi_f = list_electrode_bias[index]
                    else:
                        phi_f = list_phi_f[index_float]
                    Alpha_LBC_float  = (-1)*((3/(2*step_z)) + dSj_func(phi_f,phi_sh_init[list_Nr[index]:list_Nr[index+1]] ) )
                    data_A[list_index_LBC[index]:list_index_LBC[index]+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = Alpha_LBC_float
                    index_float += 1
                    
                if list_electrode_configuration[index] == "d":
                    Alpha_LBC = (-1)*(3/(2*step_z)) 
                    data_A[list_index_LBC[index]:list_index_LBC[index]+3*len(np.arange(Nz*list_Nr[index],Nz*list_Nr[index+1], Nz)):3] = Alpha_LBC
            #----------------------------------------------------------------
            sparse_matrix_A = spsp.csr_matrix((data_A,(row_A,col_A)),shape=(K,K))
            # matrix_A = sparse_matrix_A.todense()
            #==============================================================================
            #---------------------:Fill-BOUNDARY-CONDITIONS-VECTOR-------------------------
            #==============================================================================
            index_float = 0
            for index in np.arange(nb_electrode):
                
                if list_electrode_configuration[index] == "b":
                    phi_b = list_electrode_bias[index]
                    data_BC[list_Nr[index]:list_Nr[index+1]] = Sj_func_VC(phi_b,phi_sh_init[list_Nr[index]:list_Nr[index+1]]) - dSj_func_VC(phi_b,phi_sh_init[list_Nr[index]:list_Nr[index+1]])*phi_sh_init[list_Nr[index]:list_Nr[index+1]] 
    
                if list_electrode_configuration[index] == "f":
                    if iteration == 0:
                        phi_f = list_electrode_bias[index]
                    else:
                        phi_f = list_phi_f[index_float]
                    data_BC[list_Nr[index]:list_Nr[index+1]] = Sj_func(phi_f,phi_sh_init[list_Nr[index]:list_Nr[index+1]]) - dSj_func(phi_f,phi_sh_init[list_Nr[index]:list_Nr[index+1]])*phi_sh_init[list_Nr[index]:list_Nr[index+1]] 
                    index_float += 1
                
                if list_electrode_configuration[index] == "d":
                    data_BC[list_Nr[index]:list_Nr[index+1]] = 0
                    
            if BC_electrode_ground == "D":
                index = nb_electrode
                data_BC[list_Nr[-2]:] = (np.log(r[list_Nr[index]:list_Nr[index+1]]/rg) / np.log(r[list_Nr[-2]-1]/rg))*phi_sh_init[list_Nr[-2]-1] 
            else:
                data_BC[list_Nr[-2]:] = 0
            #--
            vect_BC_sparse = spsp.csr_matrix((data_BC,(row_BC,col_BC)),shape=(K,1))
            #vect_BC_sparse = vect_BC_sparse.todense()
            #==============================================================================
            #---------------------:LINEAR SOLVE-------------------------
            #==============================================================================
        
            phi_FDscheme = spsp.linalg.spsolve(sparse_matrix_A, vect_BC_sparse)
            #phi_FDscheme = spsp.linalg.bicgstab(sparse_matrix_A, vect_BC_sparse)
            phi_FDscheme = phi_FDscheme.reshape(Nr,Nz)
        
        
            #------------------------- SAVE 
            phi_sh_init    = phi_FDscheme[:,0]  
            j_para_sh      = sigma_para0*(1/(2*step_z))*(-3*phi_FDscheme[:,0] + 4*phi_FDscheme[:,1] - phi_FDscheme[:,2]) 
            # I_null         = np.sum(j_para_sh[Nre:Nrf]*Sr[Nre:Nrf])
            #==============================================================================
            #---------------------:FIND THE FLOATING TENSION------------------------
            #==============================================================================
            index_float = 0
            for index in np.arange(nb_electrode):
                if list_electrode_configuration[index] == "f":
                    find_phi_f  = lambda x: sum( -1*j_is*(1  - np.exp(Lambda + (x - phi_FDscheme[j,0] )/Te ) )*Sr[j] for j in np.arange(list_Nr[index],list_Nr[index+1]) )
                    #----------------------------------> BRENT <-------------------------
                    phi_f_update = brentq(find_phi_f,np.min(list_electrode_bias),-1*np.min(list_electrode_bias))
                    list_phi_f[index_float]        = phi_f_update
                    #---------------------------------->       <------------------------- 
                    list_I_null[index_float] = np.sum(  j_is*(1  - np.exp(Lambda + (phi_f - phi_FDscheme[list_Nr[index]:list_Nr[index+1],0] )/Te ) )*Sr[list_Nr[index]:list_Nr[index+1]]  )   
                    #---------------------------------->
                    index_float += 1
                #--------------------------> CONVERGENCE CRITERIA 
            if iteration > 0:
                #----> NORME L2 CRITERIA 
                err_L2 = np.sqrt(np.nansum((phi_FDscheme[1:-1,1:-1]-phi_save[1:-1,1:-1])**2/(phi_FDscheme[1:-1,1:-1]**2)))
                if (iteration % 100) == 0:
                    print(err_L2, iteration ) 
                    print("--")
                if iteration == 5000:
                    err_L2 = 1e-10
            if 'f' not in list_electrode_configuration:
                I_null = 1e-10
            else:
                I_null = np.max(np.abs(list_I_null))
            #------------------------- Save potential
            phi_save = phi_FDscheme.copy()
            #------------------------- ITERATION
            iteration +=1
        
        
        
    print("Time      -> t = ",round((time.time() - start_time),5)," s")
    print("Iteration -> I = ", iteration)
    print("--")
    print("phi_sh(0) = ", phi_FDscheme[0,0], " V")
    index_float = 0
    for index in np.arange(nb_electrode):
        if list_electrode_configuration[index] == "f":
            list_electrode_bias[index] = list_phi_f[index_float]
            print("phi_f = ", list_phi_f[index_float], " V")
            index_float += 1
    return phi_FDscheme, list_electrode_bias, list_phi_f


def Import_input_condcutivity(ne,P,Te,Ti,Tn,B):
    # =============================================================================
    # ---------------------: INPUT FILES
    # =============================================================================
    InputFolder = "input.dat"
    dict_input = {}
    list_electrode_bias = []
    list_electrode_configuration = []
    list_electrode_radius = []
    for i in open(InputFolder).readlines():
        x = i.strip().split() 
        if x[0][0] != '/':
            if x[0][0] == '0':
                nb_electrode = int(np.float64(x[2]))
            elif x[0][0] == '1': 
                list_electrode_bias.extend( [(np.float64(x[k+2])) for k in np.arange(nb_electrode) ])
            elif x[0][0] == '2': 
                list_electrode_configuration.extend( [x[k+2] for k in np.arange(nb_electrode) ])
            elif x[0][0] == '3': 
                list_electrode_radius.extend( [(np.float64(x[k+2])) for k in np.arange(nb_electrode) ]) 
            elif x[0][0] == '4':
                BC_electrode_ground = x[2] 
            else:
                dict_input[x[0]] = np.float64(x[1])
    #==============================================================================
    #---------------------:PHYSICAL-CONSTANT---------------------------------------
    #==============================================================================
    NA         = dict_input.get('NA')           # Avogadro  Number      [mol^-1] 
    kB         = dict_input.get('kB')           # Boltzmann Constant
    CoulombLog = dict_input.get('CoulombLog')   # Coulomnb  Logarithm
    eps0       = dict_input.get('eps0')         # Coulomnb  Logarithm
    sig0       = dict_input.get('sig0')         # Coulomnb  Logarithm
    e          = dict_input.get('e')            # Eletric   Charge      [C]
    me         = dict_input.get('me')           # Electron  Mass        [kg]
    #==============================================================================
    #---------------------:PLASMA-PARAMETERS---------------------------------------
    #==============================================================================
    mi_amu     = dict_input.get('mi_amu')       # Argon atom mass      [amu] 
    mi         = mi_amu*1.6605e-27              # Argon atom mass      [kg] 
    eta        = me/mi
    nN = (1/(kB*Tn))*P                          # Neutral   Density     [m^-3]
    # =============================================================================
    # ---------------------: Cyclotron frequency ----------------------------------
    # =============================================================================
    Omega_i = (e*B)/mi
    Omega_e = (e*B)/me
    # =============================================================================
    # ---------------------: Collision frequency ----------------------------------
    # =============================================================================
    nu_en = sig0*nN*np.sqrt((8*e*Te)/(np.pi*me))
    nu_in = sig0*nN*np.sqrt((8*e*Ti)/(np.pi*mi))
    nu_ei = (e**(5/2)*CoulombLog/(6*np.sqrt(2)*np.pi**(3/2)*eps0**2*np.sqrt(me)))*ne*Te**(-3/2)
    # =============================================================================
    # ---------------------: MOBILITY and DIFFUSION -------------------------------
    # =============================================================================
    mu_e_para = e/(me*(nu_ei+nu_en))
    D_e_para  = Te*mu_e_para
    #-->
    mu_e_perp = (me/(e*B**2))*((nu_ei+nu_en)/(1 + (nu_ei+nu_en)**2/Omega_e**2)) 
    D_e_perp  = Te*mu_e_perp
    #-->
    mu_i_para = e/(me*(nu_in)) 
    D_i_para  = Te*mu_i_para
    #-->
    mu_i_perp = (mi/(e*B**2))*((nu_in)/(1 + (nu_in)**2/Omega_i**2))
    D_i_perp  = Te*mu_i_perp
    # =============================================================================
    # ---------------------: CONDUCTIVITY -----------------------------------------
    # =============================================================================
    sigma_perp0 = e*ne*(mu_i_perp + mu_e_perp)
    sigma_para0 = e*ne*(mu_i_para + mu_e_para)
    mu          = (sigma_perp0/sigma_para0)
    #--------------------------------------------------------------------------
    return sigma_perp0, sigma_para0, mu_i_perp, mu_i_para, nu_in, nu_en, nu_ei, Omega_i, Omega_e