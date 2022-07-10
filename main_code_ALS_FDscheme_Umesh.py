# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:40:17 2020
@author: TROTABAS Baptiste
"""
#--> Library
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from math import floor, log10
#--> Function
from func_code_ALS_solver import *



def cm2inch(value):
    return value/2.54


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\times 10^{{{1:d}}}$".format(coeff, exponent, precision)

def sci_notation_pow(num, decimal_digits=1, precision=None, exponent=None):
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    return r"$10^{{{1:d}}}$".format(coeff, exponent, precision)

def where_better(table, element):
    for element_table in table:
        if element_table < element:
            return np.where( table <  element )[0][0]
    return -1
        
        
origin = 'lower'
list_color3 = ["dodgerblue", "darkorange"]
list_color2 = ["green", "red"]
list_color1 = ["saddlebrown", "dodgerblue", "gold","lime", "green", "teal", "navy", "mediumpurple", "orchid", "red"]
list_color = ["dodgerblue", "darkorange", "forestgreen", "black", "chocolate", "blueviolet", "gold","black",
              "gray","gold","lime","saddlebrown", "peru", "gold","lime", "green", "teal", "navy", "mediumpurple", "orchid", "red"]
list_color = ["dodgerblue", "darkorange", "black", "darkviolet", "peru", "blueviolet", "gold","black","gray","gold","lime","saddlebrown", "peru", "gold","lime", "green", "teal", "navy", "mediumpurple", "orchid", "red"]
list_linestyle = ["solid","dashed","dashdot"]
list_marker = ["o","D","s","x","d"]
list_regime = [r'Not Saturated: $\tau_P \simeq 10^{-5}$', r'Partially Saturated: $\tau_P \simeq 10^{-3}$', r'Saturated: $\tau_P \simeq 10^{-2}$', r'Saturated: $\tau_P \simeq 1$']
#-->
dict_color = {}
dict_color["b"] = "navy"
dict_color["f"] = "blueviolet"
dict_color["d"] = "dimgray"
dict_color["D"] = "chocolate"
# %%
InputFolder = "input.dat"
datContent = [i.strip().split() for i in open(InputFolder).readlines()]
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
# =============================================================================
# ---------------------: DATA  
# =============================================================================
save_data = False
save_data_name = "phi_b2=-30V"
save_data_regime = "NS"
# save_data_name = "float=No"
# =============================================================================
# ---------------------: PICTURE  
# =============================================================================
picture_phi = False
picture_Vr  = False
picture_Vz  = False
picture_S   = False
picture_phi_sheath = True
picture_j_sh_para  = False
picture_phi_E      = False
picture_drop_para  = False
picture_phi_sheath_mid = False
#==============================================================================
#---------------------:PHYSICAL-CONSTANT
#==============================================================================
NA         = dict_input.get('NA')           # Avogadro  Number      [mol^-1] 
kB         = dict_input.get('kB')           # Boltzmann Constant
CoulombLog = dict_input.get('CoulombLog')   # Coulomnb  Logarithm
eps0       = dict_input.get('eps0')         # Coulomnb  Logarithm
sig0       = dict_input.get('sig0')         # Coulomnb  Logarithm
e          = dict_input.get('e')            # Eletric   Charge      [C]
me         = dict_input.get('me')           # Electron  Mass        [kg]
#==============================================================================
#---------------------:PLASMA-PARAMETERS
#==============================================================================
mi_amu     = dict_input.get('mi_amu')       # Argon atom mass      [amu] 
mi         = mi_amu*1.6605e-27              # Argon atom mass      [kg] 
eta        = me/mi
Ti = dict_input.get('Ti')                   # Ion       Temperature [eV]
Tn = dict_input.get('Tn')                   # Room      Temperature [K]
Te = dict_input.get('T0')                   # Electron  Temperature [eV]
ne = dict_input.get('n0')                   # Electron  Density     [m^-3]
B  = dict_input.get('B')                    # Magnetic  Field       [mT]
P  = dict_input.get('P')                    # Neutral   Pressure    [Pa]
nN = (1/(kB*Tn))*P                          # Neutral   Density     [m^-3]
#==============================================================================
#---------------------:EMISSIVE-ELECTRODE
#==============================================================================
Tw    = dict_input.get('Tw')                # Heating    Temperature [K]
W     = dict_input.get('W')                 # Work       Function    [J]
j_eth = 0                                   # Thermionic Emission    [A m^2]
Xi    = 0                                   
# =============================================================================
# ---------------------: SHEATH PROPERTIES
# =============================================================================
Cs = np.sqrt((e*Te)/mi)                     # Bohm       Velocity    [m s^-1]
Lambda = np.log(np.sqrt(1/(2*np.pi*eta)))   # Sheath     Parameter
j_is = (e*ne*Cs)                             # Ion  Sat.  currend  density  [m^-3]
#==============================================================================
#---------------------:GEOMETRICAL-PARAMETERS
#==============================================================================
rg = dict_input.get('rg')                   # Column     Radius      [m]
L = dict_input.get('L')                     # Column     Length      [m]
#==============================================================================
#---------------------:MESHING-PARAMETERS-PART-1
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
#---------------------:MESHING-PARAMETERS-PART-2
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
# ---------------------: ALS SOLVER  
# =============================================================================
Xi = 0
phi_FDscheme,list_electrode_bias, list_phi_f = ALS_solver_FDscheme_Umesh(ne,P,Te,Ti,Tn,B,Xi)
# =============================================================================
# ---------------------: Determination of Transport coefficient  
# =============================================================================
sigma_perp0, sigma_para0, mu_i_perp, mu_i_para, nu_in, nu_en, nu_ei, Omega_i, Omega_e = Import_input_condcutivity(ne,P,Te,Ti,Tn,B)
# =============================================================================
# ---------------------: Determination of perpendicular electric field   
# =============================================================================
E_perp = np.zeros([Nr,Nz])
E_perp[0,:]    = (-1)*(1/(2*step_r))*(-3*phi_FDscheme[0,:] + 4*phi_FDscheme[1,:] - phi_FDscheme[2,:])
E_perp[-1,:]   = (-1)*(-1/(2*step_r))*(-3*phi_FDscheme[Nr-1,:] + 4*phi_FDscheme[Nr-2,:] - phi_FDscheme[Nr-3,:])
E_perp[1:-1,:] = (-1)*(1/(2*step_r))*(phi_FDscheme[2:,:] - phi_FDscheme[0:-2,:])
#-->
E_para = np.zeros([Nr,Nz])
E_para[:,0]    = (-1)*(1/(2*step_z))*(-3*phi_FDscheme[:,0] + 4*phi_FDscheme[:,1] - phi_FDscheme[:,2])
E_para[:,-1]   = (-1)*(-1/(2*step_z))*(-3*phi_FDscheme[:,Nz-1] + 4*phi_FDscheme[:,Nz-2] - phi_FDscheme[:,Nz-3])
E_para[:,1:-1] = (-1)*(1/(2*step_z))*(phi_FDscheme[:,2:] - phi_FDscheme[:,0:-2])
# =============================================================================
# ---------------------: Determination of current density  
# =============================================================================
j_perp = sigma_perp0*E_perp
j_para = sigma_para0*E_para
# =============================================================================
# ---------------------: Determination of velocity 
# =============================================================================
Vr = mu_i_perp*E_perp
Vz = mu_i_para*E_para
# =============================================================================
# ---------------------: Determination of surface term 
# =============================================================================
dr_Vr = np.zeros([Nr,Nz])
dr_Vr[0,:]    = 2*(1/(2*step_r))*(-3*Vr[0,:] + 4*Vr[1,:] - Vr[2,:])
dr_Vr[-1,:]   = (-1/(2*step_r))*(-3*Vr[Nr-1,:] + 4*Vr[Nr-2,:] - Vr[Nr-3,:])
dr_Vr[1:-1,:] = (1/(2*step_r))*(Vr[2:,:] - Vr[0:-2,:])
#-->
dz_Vz = np.zeros([Nr,Nz])
dz_Vz[:,0]    = (1/(2*step_z))*(-3*Vz[:,0] + 4*Vz[:,1] - Vz[:,2])
dz_Vz[:,-1]   = (-1/(2*step_z))*(-3*Vz[:,Nz-1] + 4*Vz[:,Nz-2] - Vz[:,Nz-3])
dz_Vz[:,1:-1] = (1/(2*step_z))*(Vz[:,2:] - Vz[:,0:-2])
#-->
nu_SN = np.zeros([Nr,Nz])
nu_SN[0,:]  = dr_Vr[0,:]  + dz_Vz[0,:]
nu_SN[1:,:] = dr_Vr[1:,:] + dz_Vz[1:,:]
# %%
# =============================================================================
# ---------------------: Save data
# =============================================================================
if save_data:
    np.savetxt('phi_'    + save_data_name + "_" + save_data_regime +  '.txt', phi_FDscheme)
    np.savetxt('j_perp_' + save_data_name + "_" + save_data_regime +  '.txt', j_perp)
    np.savetxt('j_para_' + save_data_name + "_" + save_data_regime +  '.txt', j_para)

# %%
r_label = [] #np.zeros(int(nb_electrode+2))
r_label.append(r'$0$')

# r_label[0] = r'$0$'
# r_label[-1] = r'$r_g$'
if list_electrode_configuration.count("b") == 1:
    for index_config, config  in enumerate(list_electrode_configuration):
        if config == "b":
            # r_label[index_config+1] = r'$r_e$' 
            r_label.append(r'$r_e$')
        elif config == "f":
            # r_label[index_config+1] = r'$r_f$' 
            r_label.append(r'$r_f$')
        elif config == "d":
            # r_label[index_config+1] = r'$r_f$' 
            r_label.append(r'$r_d$')
elif list_electrode_configuration.count("b") != 1:
    cpt_b = 1
    cpt_f = 1
    for index_config, config  in enumerate(list_electrode_configuration):
        if config == "b":
            # r_label[index_config+1] = r'$r_{e_{%s}}$' % int(cpt_b)
            r_label.append(r'$r_{e_{%s}}$' % int(cpt_b))
            cpt_b += 1
        elif config == "f":
            # r_label[index_config+1] = r'$r_f$' 
            r_label.append(r'$r_f$')
            
r_label.append(r'$r_g$')
# %%
# =============================================================================
# ---------------------: PICTURES - distribution of plasma potential
# =============================================================================
if picture_phi:
    X,Y = np.meshgrid(z,r)
    fig,ax = plt.subplots()
    fig.suptitle(r'$\textrm{Without a floating electrode}$')
    plt.gcf().subplots_adjust(right = 0.91, left = 0.15, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )      #------------------------------------
    if np.max(phi_FDscheme) > 0:
        levels_pos = np.linspace(0, np.max(phi_FDscheme),10)
        kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=0, vmax=np.max(phi_FDscheme), origin='lower')
        CF1_pos = ax.contourf(X,Y, phi_FDscheme[:,:], **kw_pos)
        cbar_pos = fig.colorbar(CF1_pos)

    if np.min(phi_FDscheme) <= 0:
        neg_min = np.min(phi_FDscheme[:,:])
        neg_max = np.max(phi_FDscheme[:,:])
        levels_neg = np.linspace(neg_min, neg_max,50)
        kw_neg = dict(levels=levels_neg, cmap="YlGnBu", vmin=neg_min, vmax=neg_max, origin='lower')       
        CF1_neg = ax.contourf(X,Y, phi_FDscheme[:,:], **kw_neg)
        #=========================> CBAR
    if (np.max(phi_FDscheme) > 0 and np.min(phi_FDscheme) < 0):
        cbar_pos = fig.colorbar(CF1_neg)
        cbar_pos.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
    if (np.max(phi_FDscheme) <= 0 and np.min(phi_FDscheme) < 0):
        cbar_neg = fig.colorbar(CF1_neg)
        cbar_neg.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
    if (np.max(phi_FDscheme) > 0 and np.min(phi_FDscheme) >= 0):
        cbar_pos = fig.colorbar(CF1_neg)
        cbar_pos.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
    
    y_position = r[list_Nr]
    my_yticks = [r'$%s$' % x for x in y_position ]
    ax.set_yticks(y_position)
    ax.set_yticklabels(my_yticks)
    ax.set_ylabel(r'$r$ $\mathrm{(m)}$')
    ax.set_xlabel(r'$z$ $\mathrm{(m)}$')
    
    # current density
    current_density = np.sqrt(j_perp**2+j_para**2)
    lw = 2.5*current_density / current_density.max()
    ax.streamplot(X, Y, j_para, j_perp, density=1, color='coral', linewidth=lw)
    
    # # (r,z) limit
    # ax.set_xlim([-1,0])
    # ax.set_ylim([0, 1])
    ax.set_xlim([-L/2,0])
    ax.set_ylim([0, rg])


# %%
# =============================================================================
# ---------------------: PICTURES - distribution of ion radial velocity
# =============================================================================
if picture_Vr:
    #------------------------------------
    fig,ax = plt.subplots()
    plt.gcf().subplots_adjust(right = 0.94, left = 0.15, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
    V_max = np.max(Vr[:-1,:])
    V_min = np.min(Vr[:-1,:])
    if V_max > 0:
        pos_min = 0
        pos_max = V_max
        levels_pos = np.linspace(pos_min, pos_max,10)
        kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=pos_min, vmax=pos_max, origin='lower')
        CF1_pos = ax.contourf(X,Y, Vr[:,:], **kw_pos)
    if V_min < 0:
        neg_min = V_min
        neg_max = 0
        levels_neg = np.linspace(neg_min, neg_max,10)
        kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
        CF1_neg = ax.contourf(X,Y, Vr[:,:], **kw_neg)
    if (V_max > 0 and V_min < 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $Vr(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
        cbar_neg = fig.colorbar(CF1_neg)

    elif (V_max <= 0 and V_min < 0):
        cbar_neg = fig.colorbar(CF1_neg)
        cbar_neg.set_label(r'  $Vr(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    elif (V_max > 0 and V_min >= 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $Vr(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    ax.set_ylabel(r'$r/r_e$')
    ax.set_xlabel(r'$z/[L/2]$')

# %%
# =============================================================================
# ---------------------: PICTURES - distribution of ion axial velocity
# =============================================================================
if picture_Vz:
    fig,ax = plt.subplots()
    plt.gcf().subplots_adjust(right = 0.94, left = 0.15, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
    if np.max(Vz[:,:]) > 0:
        pos_min = 0
        pos_max = np.max(Vz[:,:])
        levels_pos = np.linspace(pos_min, pos_max,10)
        kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=pos_min, vmax=pos_max, origin='lower')
        CF1_pos = ax.contourf(X,Y, Vz[:,:], **kw_pos)
    if np.min(Vz[:,:]) < 0:
        neg_min = np.min(Vz[:,:])
        neg_max = 0
        levels_neg = np.linspace(neg_min, neg_max,10)
        kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
        CF1_neg = ax.contourf(X,Y, Vz[:,:], **kw_neg)
    if (np.max(Vz[:,:]) > 0 and np.min(Vz[:,:]) < 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $Vz(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
        cbar_neg = fig.colorbar(CF1_neg)
    elif (np.max(Vz[:,:]) <= 0 and np.min(Vz[:,:]) < 0):
        cbar_neg = fig.colorbar(CF1_neg)
        cbar_neg.set_label(r'  $Vz(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    elif (np.max(Vz[:,:]) > 0 and np.min(Vz[:,:]) >= 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $Vz(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    ax.set_ylabel(r'$r/r_e$')
    ax.set_xlabel(r'$z/[L/2]$')
    
# %%
# =============================================================================
# ---------------------: PICTURES - distribution of  source term
# =============================================================================
Src = np.abs(nu_SN/nu_in)
if picture_S:
    levels_log = np.logspace( np.log10(np.max(Src*1e-2)), np.log10(np.max(Src)),10)
    kw = dict(levels=levels_log, locator=ticker.LogLocator(), cmap=cm.YlGnBu )
    
    fig,ax = plt.subplots()
    plt.gcf().subplots_adjust(right = 0.75, left = 0.15, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
    CF1 = ax.contourf(X,Y, Src[:,:], **kw)
    #=========================> CBAR
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.yaxis.set_major_locator(ticker.NullLocator())
    
    cbar = fig.colorbar(CF1, cax = cax) #, cax = cax, ticks = levels)
    cbar.set_ticks([])
    loc    = levels_log 
    labels = [ sci_notation(levels_log[i]) for i in range(len(levels_log)) ]

    cbar.set_ticks(loc)
    cbar.set_ticklabels(labels)
    cbar.set_label(r'  $S(r,z)$ $\mathrm{(m^{-3} \cdot s^{-1})}$', rotation = 90)
    ax.set_ylabel(r'$r/r_e$')
    ax.set_xlabel(r'$z/[L/2]$')

# %%
# =============================================================================
# ---------------------: PICTURES - radial profile : phi + j
# =============================================================================
if picture_phi_sheath:
    fig, ax = plt.subplots(constrained_layout = False) 
    plt.gcf().subplots_adjust(right = 0.85, left = 0.15, wspace = 0.25, top = 0.98, hspace = 0.1, bottom = 0.14 )
    # fig.suptitle(r'$\textrm{Electron temperature: Case } %s$, $\phi_b = %s \mathrm{V}$, $Xi = %s$, $<T_e>_{cath} = %s \mathrm{eV}$ $(%s/4)$' % (InputCase,ElectrodeBias, round(Xi,1), round(np.mean(Te_f(r[:Nre])),1), int(HeatingCurrent)) )
    label1 = ax.plot(r, phi_FDscheme[:,0], color = "black", linestyle = "solid", label = r'$\phi_{sh}(r)$')
    
    for index_phi_b, phi_b in enumerate(list_electrode_bias):
        phi_NS = phi_b + Lambda*Te
        # ax.axhline(y = phi_NS, color = "black", linestyle = "dotted")
        # ax.text(0.135, phi_NS+2, r'$\phi_{NS}^{(%s)} = %s$ $\mathrm{V}$' % (int(index_phi_b+1), int(phi_NS)) )
    #-->
    ax2 = ax.twinx()
    label2 = ax2.plot(r, j_para[:,0]/j_is, color = "crimson", linestyle = "solid", label = r'$j_{sh,\parallel}(r)/j_{is}$')
    ax2.axhline(y=0, color = "crimson", linestyle = "dotted")
    # added these three lines
    label = label1  + label2
    labs = [l.get_label() for l in label]
    ax.legend(label, labs, loc=0, frameon = False)
    #-->
    x_position = r[list_Nr]
    my_xticks = [r'$%s$' % x for x in x_position ]
    ax.set_xticks(x_position)
    ax.set_xticklabels(r_label)
    ax.set_xlabel(r'$\textrm{Plasma radius}$')
    ax.set_ylabel(r'$\phi_{sh}(r)$ $\mathrm{(V)}$')
    ax2.set_ylabel(r'$j_{sh,\parallel}(r)/j_{is}$')
    
    


# %%
# =============================================================================
# ---------------------: PICTURES - radial profile : phi + electric field
# =============================================================================
if picture_phi_E:
    fig, ax = plt.subplots(constrained_layout = False) 
    plt.gcf().subplots_adjust(right = 0.85, left = 0.135, wspace = 0.25, top = 0.99, hspace = 0.1, bottom = 0.14 )
    label1 = ax.plot(r, phi_FDscheme[:,0], color = "black", linestyle = "solid", label = r'$\phi_{sh}(r)$')
    # ax.plot(r_m*1e2, phi_m_VKP, color = "crimson", linestyle = "None", marker = marker_type, markersize = marker_size)#, label = r'$\textrm{measurement}$')
    ax2 = ax.twinx()
    # label3 = ax2.plot(r_v*1e2, E_perp_VKP, color = "navy", linestyle = "solid", label = r'$E_\perp(r,z_0)$') 
    label2 = ax2.plot(r, E_perp[:,0], color = "navy", linestyle = "solid", label = r'$E_\perp(r,z_m)$') 
    ax2.axhline(y = 0, color = "navy", linestyle = "dashed")
    ax2.set_ylabel(r'$E_\perp(r,z_m)$ $\mathrm{(V \cdot m^{-1})}$  ')
    # added these three lines
    label = label1  + label2
    labs = [l.get_label() for l in label]
    ax.legend(label, labs, loc=0, frameon = False)

    ax.set_xticks(x_position)
    ax.set_xticklabels(my_xticks)
    ax.set_xlabel(r'$\textrm{Plasma radius}$')
    ax.set_ylabel(r'$\phi_{sh}(r)$ $\mathrm{(V)}$')
    
    
# %%
# =============================================================================
# ---------------------: PICTURES - Voltage drop along field lines
# =============================================================================
if picture_drop_para:
    fig, ax = plt.subplots(constrained_layout = False) 
    plt.gcf().subplots_adjust(right = 0.99, left = 0.18, wspace = 0.25, top = 0.99, hspace = 0.1, bottom = 0.14 )
    # fig.suptitle(r'$\textrm{Electron temperature: Case } %s$, $\phi_b = %s \mathrm{V}$, $Xi = %s$, $<T_e>_{cath} = %s \mathrm{eV}$ $(%s/4)$' % (InputCase,ElectrodeBias, round(Xi,1), round(np.mean(Te_f(r[:Nre])),1), int(HeatingCurrent)) )
    delta_para_psi = (phi_FDscheme[:,-1] - phi_FDscheme[:,0])/Te
    delta_para_psi_th = -1*(L/(4*Te*sigma_para0))*j_para[:,0]
    ax.plot(r, delta_para_psi, color = "black", linestyle = "solid", label = r'$\textrm{Simulation}$')
    ax.plot(r, delta_para_psi_th, color = "crimson", linestyle = "dotted", label = r'$\textrm{Theory}$')

    ax.legend(loc=0, frameon = False)
    #-->
    x_position = r[list_Nr]
    my_xticks = [r'$%s$' % x for x in x_position ]
    ax.set_xticks(x_position)
    ax.set_xticklabels(my_xticks)
    ax.set_xlabel(r'$\textrm{Plasma radius}$')
    ax.set_ylabel(r'$\Delta_\parallel \psi(r)$')

