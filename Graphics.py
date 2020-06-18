# -*- coding: utf-8 -*-
"""
######################################################
#           Plotting script for journal              #
#              Created by J. Mark Epps               #
#          Part of Masters Thesis at UW 2018-2020    #
######################################################

mark_epps@hotmail.ca

This file is the script used to produce the figures 
based on the simulation data for Al/CuO nanothermite 
pellet combustion. 

    
Features:
    -curve fitting routine to obtain fitting parameters (log-log or log-linear)
    -plot variables from 'Post-processing.txt' files for each simulation
    -For each boolean variable in 'Options for figures' section represents separate figure
    -For 'all_cases' output for a given set of axes, a figure with 2 subplots is generated
    -For 'all_cases' output, commented sections was for 2 separate figures

"""

import os
import matplotlib as mtplt
import string as st
from matplotlib import pyplot as plt
#from matplotlib import ticker, cm
import numpy as np
import csv
from scipy.optimize import curve_fit
from myFigs import set_size


# Function to collect info
def get_data():
    # Create empty dictionary
    data_dict={
          'porosity': [],
          'ign' : [],
          'v_avg': [],
          'Pe': [],
          'Da': [],
          'nd_ign' : [],
          'nd_v_avg': [],
          'adv':[],
          'u_ref': []
              }
    for root, dirs, files in os.walk(".", topdown=False):
        for folders in dirs:
            i=0 # Info counter
            try:
                fin=open(os.path.join(root, folders)+'\\Input_file.txt')
                line=' '
                while i<4 and line!='':
                    line=fin.readline()
                    if st.find(line, 'Porosity')>=0:
                        data_dict['porosity'].append(100-100*float(st.split(line, ':')[1]))
                        i+=1

                    if st.find(line, 'k_s')>=0:
                        k_s=float(st.split(line, ':')[1])
                        i+=1
                    
                    if st.find(line, 'Average')>=0:
                        line_sub=st.split(st.split(line, ':')[1], 'm')[0]
                        data_dict['v_avg'].append(float(line_sub))
                        i+=1
                        
                    if st.find(line, 'Ignition time')>=0:
                        line_sub=st.split(st.split(line, ':')[1], 'm')[0]
                        data_dict['ign'].append(float(line_sub))
                        i+=1
                
                fin.close()
                try:
                    fin=open(os.path.join(root, folders)+'\\Post_processing_Cp.txt')
                    while i<9 and line!='':
                        line=fin.readline()
                        if st.find(line, 'Pe_y')>=0:
#                            data_dict['Pe'].append(np.log(float(st.split(line, ':')[1])))
                            data_dict['Pe'].append(float(st.split(line, ':')[1]))
                            i+=1
                        if st.find(line, 'u_ref')>=0:
#                            data_dict['Pe'].append(np.log(float(st.split(line, ':')[1])))
                            data_dict['u_ref'].append(float(st.split(line, ':')[1]))
                            i+=1
                        if st.find(line, 'Da_y')>=0:
                            data_dict['Da'].append(float(st.split(line, ':')[1]))
                            i+=1
                        if st.find(line, 'Burn')>=0:
#                            data_dict['v_avg'].append(np.log(float(st.split(line, ':')[1])))
                            data_dict['nd_v_avg'].append(float(st.split(line, ':')[1]))
                            i+=1
                            
                        if st.find(line, 'Ignition')>=0:
#                            data_dict['ign'].append(np.log(float(st.split(line, ':')[1])))
                            data_dict['nd_ign'].append(float(st.split(line, ':')[1]))
                            i+=1
                    
                    data_dict['adv'].append(k_s*data_dict['Pe'][-1])
                    fin.close()
                except:
                    data_dict['Pe'].append(0)
                    data_dict['Da'].append(0)
                    data_dict['nd_v_avg'].append(0)
                    data_dict['nd_ign'].append(0)
                    data_dict['u_ref'].append(0)
            
            except:
                continue
    return data_dict

# Function to mesh case data
def get_data_mesh(data_dict={
          'porosity': [],
          'ign' : [],
          'v_avg': [],
          'Pe': [],
          'Da': [],
          'nd_ign' : [],
          'nd_v_avg': [],
          'pressure': []
              }):
    fin=open('Input_file.txt')
    line=' '
    len_x,len_y,i=0,0,0
    while i<4 and line!='':
        line=fin.readline()
        if st.find(line, 'Nodes_x')>=0:
            len_x=int(st.split(line, ':')[1])
            i+=1
        if st.find(line, 'Nodes_y')>=0:
            len_y=int(st.split(line, ':')[1])
            i+=1
        if st.find(line, 'Average')>=0:
            line_sub=st.split(st.split(line, ':')[1], 'm')[0]
            data_dict['v_avg'].append(float(line_sub))
            i+=1
        if st.find(line, 'Ignition time')>=0:
            line_sub=st.split(st.split(line, ':')[1], 'm')[0]
            data_dict['ign'].append(float(line_sub))
            i+=1
    fin.close()
    data_dict['porosity'].append(len_x*len_y)
    
    try:
        fin=open('Post_processing.txt')
        line=' '
        while i<9 and line!='':
            line=fin.readline()
            if st.find(line, 'Pe_y')>=0:
#                data_dict['Pe'].append(np.log(float(st.split(line, ':')[1])))
                data_dict['Pe'].append(float(st.split(line, ':')[1]))
                i+=1
            if st.find(line, 'Da_y')>=0:
                data_dict['Da'].append(float(st.split(line, ':')[1]))
                i+=1
            if st.find(line, 'Burn')>=0:
#                data_dict['v_avg'].append(np.log(float(st.split(line, ':')[1])))
                data_dict['nd_v_avg'].append(float(st.split(line, ':')[1]))
                i+=1
                
            if st.find(line, 'Ignition')>=0:
#                data_dict['ign'].append(np.log(float(st.split(line, ':')[1])))
                data_dict['nd_ign'].append(float(st.split(line, ':')[1]))
                i+=1
            if st.find(line, 'pressure')>=0:
                data_dict['pressure'].append(float(st.split(line, ':')[1]))
                i+=1
            
        fin.close()
    except:
        data_dict['Pe'].append(0)
        data_dict['Da'].append(0)
        data_dict['nd_v_avg'].append(0)
        data_dict['nd_ign'].append(0)
        data_dict['pressure'].append(0)

    return data_dict

# Function to calculate effective thermal conductivity via EMT model
def emt(ks,kf,porosity):
    import sympy as sp
    from sympy.utilities.lambdify import lambdify
    x = sp.symbols('x')
    k_emt=np.zeros_like(porosity)
    j=0
    for pore in porosity:
        y=(1-pore)*(ks-x)/(ks+2*x)+pore*(kf-x)/(kf+2*x)
        if j==0:
            x0=ks # Initial guess for zero
            x1=ks
        else:
            x0=k_emt[j-1] # Initial guess for zero
            x1=k_emt[j-1]
        conv=0.0001 # Convergence criteria
        max_iter=1000 # Max iterations to find solution
        dy=y.diff(x)
        # Create callable function
        f=lambdify(x,y)
        df=lambdify(x,dy)
        i=1
        # Find zero
        while i==1 or (abs(x0-x1)/x0>conv and i<max_iter):
            x1=x0
            x0=x0-f(x0)/df(x0)
            i+=1
        k_emt[j]=x0
        j+=1
    return k_emt

# Function to extract transient BR data from an input file
def get_BR_data():
    t=[]
    BR=[]
    fin=open('Input_file.txt', 'r')
    for line in fin:
        if st.find(line, 'inst')>=0:
            # Save time
            time=st.split(line, 't=')[1]
            time=st.split(time, 'ms')[0]
            t.append(float(time))
            # Save BR
            br=st.split(line, 'inst-')[1]
            br=st.split(br, ',')[0]
            BR.append(float(br))
        if st.find(line, 'Ignition time')>=0:
            t0=float(st.split(st.split(line, ':')[1], 'm')[0])
    t1=max(t)
    for i in range(len(t)):
        t[i]=(t[i]-t0)/(t1-t0)
    
    return [t,BR]

# Curve fitting functions
def func(x,a,b):
#    return a*np.exp(b*x)
    return b*x+a
def func_quad(x,a,b,c):
    return a*x**2+b*x+c

def curve_fit_res(x,y,fit_type):
    # Regression slopes
    if fit_type=='log-log':
        a,dum=curve_fit(func, np.log(x), np.log(y), p0=(4*10**(-6), -0.2))
    elif fit_type=='log-linear':
        a,dum=curve_fit(func, x, np.log(y), p0=(4*10**(-6), -0.2))
    else:
        a,dum=curve_fit(func_quad, x, y)
    
    # Residuals (https://en.wikipedia.org/wiki/Coefficient_of_determination)
    y_avg=0
    for i in range(len(x)):
        y_avg+=y[i]
    y_avg/=len(x)
    
    SS_tot=0
    SS_res=0
    for i in range(len(x)):
        SS_tot+=(y[i]-y_avg)**2
        if fit_type=='log-log':
            SS_res+=(y[i]-np.exp(func(np.log(x[i]), a[0], a[1])))**2
        elif fit_type=='log-linear':
            SS_res+=(y[i]-np.exp(func(x[i], a[0], a[1])))**2
        else:
            SS_res+=(y[i]-func_quad(x[i], a[0], a[1], a[2]))**2
    
    R=1-SS_res/SS_tot
#    a.append(R)
    return a,R

def interpolate(k1, k2, func):
    if func=='Linear':
        return 0.5*k1+0.5*k2
    else:
        return 2*k1*k2/(k1+k2)

plt.ioff()

print('######################################################')
print('#           Plotting script for journal              #')
print('#              Created by J. Mark Epps               #')
print('#          Part of Masters Thesis at UW 2018-2020    #')
print('######################################################\n')

##############################################################
#               Collect Data into dictionaries
##############################################################
# base data
os.chdir('1_base')
base=get_data()
base['legend']='base'#, '$E_a$=48 kJ/mol', '$d^*$=3.96']
base['plt']=['^','black','full']

# dc data
os.chdir('..\\dc\\1um')
dc_1um=get_data()
dc_1um['legend']='$d_c=1\: \mu m$'
dc_1um['plt']=['s','blue','full']
os.chdir('..\\100nm')
dc_100nm=get_data()
dc_100nm['legend']='$d_c=100\: nm$'
dc_100nm['plt']=['s','blue','none']

# Cp data
os.chdir('..\\..\\gas\\Al')
Cp_Al=get_data()
Cp_Al['legend']='$Al_{(g)}$'
Cp_Al['plt']=['o','red','full']
os.chdir('..\\Cu')
Cp_Cu=get_data()
Cp_Cu['legend']='$Cu_{(g)}$'
Cp_Cu['plt']=['o','red','none']

# lambda data
os.chdir('..\\..\\lambda\\1W')
lam_1W=get_data()
#lam_1W['legend']='$\lambda_{eff}=1\: W\,m^{-1}K^{-1}$'
lam_1W['legend']='$\lambda_{eff}=1$'
lam_1W['plt']=['D','orange','none']
os.chdir('..\\100W')
lam_100W=get_data()
#lam_100W['legend']='$\lambda_{eff}=100\: W\,m^{-1}K^{-1}$'
lam_100W['legend']='$\lambda_{eff}=100$'
lam_100W['plt']=['D','orange','full']

# mu data
os.chdir('..\\..\\mu\\4Pa')
mu_4=get_data()
#mu_4['legend']='$\mu=10^{-4}\: kg\, m^{-1} s^{-1}$'
mu_4['legend']='$\mu=10^{-4}$'
mu_4['plt']=['P','green','none']
os.chdir('..\\3Pa')
mu_3=get_data()
#mu_3['legend']='$\mu=10^{-3}\: kg \, m^{-1} s^{-1}$'
mu_3['legend']='$\mu=10^{-3}$'
mu_3['plt']=['P','green','full']

# A0 data
os.chdir('..\\..\\A0\\489e5')
A0_489e5=get_data()
#A0_489e5['legend']='$A_0=4.89e5\: s^{-1}$'
A0_489e5['legend']='$A_0=4.89e5$'
A0_489e5['plt']=['*','cyan','none']
os.chdir('..\\244e6')
A0_244e6=get_data()
#A0_244e6['legend']='$A_0=2.44e6\, s^{-1}$'
A0_244e6['legend']='$A_0=2.44e6$'
A0_244e6['plt']=['*','cyan','full']

# Ea data
os.chdir('..\\..\\Ea\\70kJ')
Ea_70kJ=get_data()
#Ea_70kJ['legend']='$E_a=70\: kJ\, mol^{-1}$'
Ea_70kJ['legend']='$E_a=70$'
Ea_70kJ['plt']=['v','magenta','none']
os.chdir('..\\80kJ')
Ea_80kJ=get_data()
#Ea_80kJ['legend']='$E_a=80\: kJ\, mol^{-1}$'
Ea_80kJ['legend']='$E_a=80$'
Ea_80kJ['plt']=['v','magenta','full']

# Inst BR data
os.chdir('..\\..\\1_base\\1')
#os.chdir('..\\..\\lambda\\1W\\1')
BR_t0=get_BR_data()
BR_t0.append('90%TMD')
os.chdir('..\\2')
BR_t1=get_BR_data()
BR_t1.append('70%TMD')
os.chdir('..\\3')
BR_t2=get_BR_data()
BR_t2.append('50%TMD')
os.chdir('..\\4')
BR_t3=get_BR_data()
BR_t3.append('30%TMD')
os.chdir('..\\5')
BR_t4=get_BR_data()
BR_t4.append('10%TMD')

# Mesh data
#os.chdir('..\\..\\dc\\1um\\2') # OLD DIRECTORY FOR 6e4
os.chdir('..\\..\\mesh\\1um\\2')
mesh=get_data_mesh()
#BR_t0=get_BR_data()
#BR_t0.append('6e4 nodes')
#os.chdir('..\\..\\..\\..\\mesh\\1um\\2') # WHEN USING OLD DIRECTORY
os.chdir('..\\2_1')
mesh=get_data_mesh(mesh)
#BR_t2=get_BR_data()
#BR_t2.append('2.4e5')
os.chdir('..\\2_2')
mesh=get_data_mesh(mesh)
#BR_t1=get_BR_data()
#BR_t1.append('1.2e5')
os.chdir('..\\2_3')
mesh=get_data_mesh(mesh)
#BR_t3=get_BR_data()
#BR_t3.append('3e5')
os.chdir('..\\..\\..') # Return to directory containing this file

# Dict with ALL data (except mesh)
compiled_data={'base': base,
      'dc_1um': dc_1um, 'dc_100nm': dc_100nm,
      'Cp_Al': Cp_Al, 'Cp_Cu': Cp_Cu,
      'lam_100W': lam_100W, 'lam_1W': lam_1W,
      'mu_3': mu_3,'mu_4': mu_4,
      'A0_244e6': A0_244e6, 'A0_489e5': A0_489e5,
      'Ea_70kJ': Ea_70kJ,'Ea_80kJ': Ea_80kJ}
data_keys_ALL=['base','dc_1um','dc_100nm','Cp_Al',
                    'Cp_Cu','lam_100W','lam_1W',
                    'mu_3','mu_4','A0_244e6','A0_489e5',
                    'Ea_80kJ','Ea_70kJ']
data_keys_trans=['base','dc_1um','dc_100nm',
                 'Cp_Al','Cp_Cu',
                 'lam_100W','lam_1W',
                 'mu_3','mu_4']
data_keys_kin=['Ea_80kJ','Ea_70kJ','A0_244e6','A0_489e5']
##############################################################
#               Options for figures
##############################################################
figType='.pdf' # Extension of figure files
width = 384
#width = 460
folder_output='Graphics' # Folder to output figures to; relative to where the data is
send_to_csv=False # UNTESTED OPTION TO SEND ALL DATA TO CSV FILE; LEAVE AS IS
is_larger_figs=False # SPECIFIC FOR THESIS SEMINAR; LEAVE AS IS

all_cases=True # Plot all simulation data
certain_cases=False # Plot  specific simulation data (specify in list below)
certain_cases_plot=['base',
                    'dc_1um','dc_100nm',
#                    'Cp_Al','Cp_Cu',
                    'lam_100W',
                    'lam_1W',
                    'mu_3','mu_4',
#                    'A0_244e6',
#                    'A0_489e5',
#                    'Ea_80kJ',
#                    'Ea_70kJ'
                    ]
legend_colns=3 # Number of columns for data in legend (certain_cases=True)
#y_axis_quant=['ign']
#x_axis_quant='porosity'
y_axis_quant=['v_avg'] # Y-axis data of figure
x_axis_quant='Pe' # x-axis data of figure
curve_fitting_routine=False # GET CURVE FIT PARAMETERS FOR X AND Y DATA SPECIFIED AND PLOT REGRESSION ON FIGURE
#['v_avg','ign','nd_v_avg','nd_ign','Pe','Da','porosity','adv','u_ref'] # Available data to plot

mesh_study=False # Plot mesh study results (separate figure from previous options)
mesh_quant='pressure' # Quantity to plot for mesh study

exp_comp=False # Plot experimental and numerical results (separate figure from previous options)

therm_cond_models=False # Plot thermal conductivity models for comparison (separate figure from previous options)
therm_cond_model_zoom=False # Previous graph with zoom in on a section (separate figure from previous options)

inst_BR_time=False # Plot instanteous burn rate data; BASELINE CASES ARE USED IN PLOTTING

is_v_contours=False # Plot axial Darcy velocity contours; BASELINE 10% AND 90% TMD CASES
is_P_trans=False # Plot transient pressure and temperature data; BASELINE 10% TMD

# Curve fitting
y_curve=y_axis_quant[0]
x_curve=x_axis_quant

y_logscale=['nd_v_avg','Da','nd_ign','Pe','u_ref']
x_logscale=['Pe','Da','u_ref']
alph_low='none'#0.3
alph_high='full'#0.8
##############################################################
#               Figure details
##############################################################
#if is_larger_figs:
#    width*=1.5
fig_size=set_size(width)
#fig_size=(6,6)
if is_larger_figs:
    nice_fonts = {
            # Use LaTex to write all text
    #        "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 16,
            "font.size": 16,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            # Default markersize is 6
            "lines.markersize": 6
    }
else:
    nice_fonts = {
            # Use LaTex to write all text
    #        "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 11,
            "font.size": 11,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            # Default markersize is 6
            "lines.markersize": 6
    }
mtplt.rcParams.update(nice_fonts)
#cmap_choice=mtplt.cm.viridis
axis_labels={
        'v_avg': '$\\bar V_{cw}$ [$ms^{-1}$]',
        'nd_v_avg': '$V_{cw}^*$ [-]',
        'ign': '$t_{ign}$ [$ms$]',
        'nd_ign': '$t_{ign}^*$ [-]',
#        'Pe': '$ln(Pe)$ [-]',
        'Pe': '$Pe$ [-]',
        'Da': '$Da$ [-]',
        'porosity': '% TMD',
        'pressure': 'Peak Pressure [Pa]',
        'adv': '$\rho_{ref}U_{ref}C_pL_{ref}$',
        'u_ref': '$U_{ref}$ [$ms^{-1}$]'
        }
titles={
        'v_avg': 'Burn rate',
        'ign': 'Ignition delay',
        'nd_v_avg': 'Burn rate',
        'nd_ign': 'Ignition delay',
        'Pe': 'Peclet',
        'Da': 'Damkohler',
        'porosity': '%TMD',
        'adv': 'Advection',
        'u_ref': 'Reference velocity'
        }
file_name={
        'v_avg': 'BR',
        'ign': 'ign',
        'nd_v_avg': 'ndBR',
        'nd_ign': 'ndign',
        'Pe': 'Pe',
        'Da': 'Da',
        'porosity': 'TMD',
        'adv': 'adv',
        'u_ref': 'uref'
        }
markers={
        'base': '^',
        'Cp': 'o',
        'dc': 's',
        'lam': 'D',
        'mu': 'x',
        'mu2': 'P',
        'Ea': 'v',
        'A0': '*'
        }
##############################################################
#               Output graphs
##############################################################
os.chdir(folder_output)
if is_larger_figs:
    os.chdir('Larger')

########################## Send data to CSV
if send_to_csv:
    with open('All_data.csv', mode='w') as csv_file:
        headers = dc_1um.keys()
        coln=len(dc_1um[headers[0]])
        headers.append(headers.pop(headers.index('legend')))
        headers.reverse()
        data=[base,dc_1um,dc_100nm,Cp_Al,Cp_Cu,lam_100W,lam_1W,mu_3,mu_4,
              A0_244e6,A0_489e5,Ea_70kJ,Ea_80kJ]
        
        # Open CSV file
        writer = csv.writer(csv_file)
#        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        
#        writer.writeheader()
        writer.writerow(headers)
        # All dictionaries
        for i in range(len(data)):
            for j in range(coln):
                dat_row=[]
                for k in range(len(headers)):
                    dat_row.append(data[i][headers[k]][j])
                writer.writerow(dat_row)
            break

########################## Curve fitting
if curve_fitting_routine:
    # log-log
    if (y_curve in y_logscale) and (x_curve in x_logscale):
        print('\n\nlog('+axis_labels[y_curve]+')=a+b*log('+axis_labels[x_curve]+')')
        # Baseline
        a_base,R=curve_fit_res(base[x_curve],base[y_curve],'log-log')
        print('base: '+str(a_base)+', R^2 = '+str(R))
        # dc - 100nm
        a,R=curve_fit_res(dc_100nm[x_curve],dc_100nm[y_curve],'log-log')
        print('100nm:'+str(a)+', R^2 = '+str(R))
        # dc - 1um
        a_dc,R=curve_fit_res(dc_1um[x_curve],dc_1um[y_curve],'log-log')
        print('d_1um:'+str(a_dc)+', R^2 = '+str(R))
        # gas - Al
        a,R=curve_fit_res(Cp_Al[x_curve],Cp_Al[y_curve],'log-log')
        print('Cp_Al:'+str(a)+', R^2 = '+str(R))
        # gas - Cu
        a,R=curve_fit_res(Cp_Cu[x_curve],Cp_Cu[y_curve],'log-log')
        print('Cp_Cu:'+str(a)+', R^2 = '+str(R))
        # lambda - 1W
        a,R=curve_fit_res(lam_1W[x_curve],lam_1W[y_curve],'log-log')
        print('1W:   '+str(a)+', R^2 = '+str(R))
        # lambda - 100W
        a,R=curve_fit_res(lam_100W[x_curve],lam_100W[y_curve],'log-log')
        print('100W: '+str(a)+', R^2 = '+str(R))
        # mu - 4Pa
        a,R=curve_fit_res(mu_4[x_curve],mu_4[y_curve],'log-log')
        print('mu_4: '+str(a)+', R^2 = '+str(R))
        # mu - 3Pa
        a_mu,R=curve_fit_res(mu_3[x_curve],mu_3[y_curve],'log-log')
        print('mu_3: '+str(a_mu)+', R^2 = '+str(R))
        # Ea - 70kJ
        a,R=curve_fit_res(Ea_70kJ[x_curve],Ea_70kJ[y_curve],'log-log')
        print('70kJ: '+str(a)+', R^2 = '+str(R))
        # Ea - 80kJ
        a,R=curve_fit_res(Ea_80kJ[x_curve],Ea_80kJ[y_curve],'log-log')
        print('80kJ: '+str(a)+', R^2 = '+str(R))
        # A0 - 244e6
        a,R=curve_fit_res(A0_244e6[x_curve],A0_244e6[y_curve],'log-log')
        print('244e6:'+str(a)+', R^2 = '+str(R))
        # A0 - 489e5
        a,R=curve_fit_res(A0_489e5[x_curve],A0_489e5[y_curve],'log-log')
        print('489e5:'+str(a)+', R^2 = '+str(R))
    
    # log-linear
    else:
        print('log('+axis_labels[y_curve]+')=a+b*'+axis_labels[x_curve])
        # Baseline
        a_base,R=curve_fit_res(base[x_curve],base[y_curve],'log-linear')
        print('base: '+str(a_base)+', R^2 = '+str(R))
        # dc - 100nm
        a,R=curve_fit_res(dc_100nm[x_curve],dc_100nm[y_curve],'log-linear')
        print('100nm:'+str(a)+', R^2 = '+str(R))
        # dc - 1um
        a_dc,R=curve_fit_res(dc_1um[x_curve],dc_1um[y_curve],'log-linear')
        print('d_1um:'+str(a_dc)+', R^2 = '+str(R))
        # gas - Al
        a,R=curve_fit_res(Cp_Al[x_curve],Cp_Al[y_curve],'log-linear')
        print('Cp_Al:'+str(a)+', R^2 = '+str(R))
        # gas - Cu
        a,R=curve_fit_res(Cp_Cu[x_curve],Cp_Cu[y_curve],'log-linear')
        print('Cp_Cu:'+str(a)+', R^2 = '+str(R))
        # lambda - 1W
        a,R=curve_fit_res(lam_1W[x_curve],lam_1W[y_curve],'log-linear')
        print('1W:   '+str(a)+', R^2 = '+str(R))
        # lambda - 100W
        a,R=curve_fit_res(lam_100W[x_curve],lam_100W[y_curve],'log-linear')
        print('100W: '+str(a)+', R^2 = '+str(R))
        # mu - 4Pa
        a,R=curve_fit_res(mu_4[x_curve],mu_4[y_curve],'log-linear')
        print('mu_4: '+str(a)+', R^2 = '+str(R))
        # mu - 3Pa
        a_mu,R=curve_fit_res(mu_3[x_curve],mu_3[y_curve],'log-linear')
        print('mu_3: '+str(a_mu)+', R^2 = '+str(R))
        # Ea - 70kJ
        a,R=curve_fit_res(Ea_70kJ[x_curve],Ea_70kJ[y_curve],'log-linear')
        print('70kJ: '+str(a)+', R^2 = '+str(R))
        # Ea - 80kJ
        a,R=curve_fit_res(Ea_80kJ[x_curve],Ea_80kJ[y_curve],'log-linear')
        print('80kJ: '+str(a)+', R^2 = '+str(R))
        # A0 - 244e6
        a,R=curve_fit_res(A0_244e6[x_curve],A0_244e6[y_curve],'log-linear')
        print('244e6:'+str(a)+', R^2 = '+str(R))
        # A0 - 489e5
        a,R=curve_fit_res(A0_489e5[x_curve],A0_489e5[y_curve],'log-linear')
        print('489e5:'+str(a)+', R^2 = '+str(R))
    
    # Testing
#    y_fit=[]
#    for i in range(len(x)):
#        y_fit.append(np.exp(func(x[i], a[0], a[1])))
#    
#    plt.scatter(x, y, label='Actual')
#    plt.plot(x, y_fit, label='Fit')
#    plt.yscale('log')
#    plt.legend()
#    plt.show()

########################## Certain cases
if certain_cases:
    for i in y_axis_quant:
        x_min,x_max,y_min,y_max=10**6,0,10**6,0
        fig=plt.figure(figsize=fig_size)
        mark_sizes={'base': 6,'dc_1um': 6,'dc_100nm':6,
                    'Cp_Al': 6,'Cp_Cu': 6,
                    'lam_100W': 4,'lam_1W': 7,
                    'mu_3': 6,'mu_4': 6,
                    'A0_244e6': 6,'A0_489e5': 6,
                    'Ea_80kJ': 6,'Ea_70kJ': 6
                }
        for case in certain_cases_plot:
            x_min=min(x_min,min(compiled_data[case][x_axis_quant]))
            x_max=max(x_max,max(compiled_data[case][x_axis_quant]))
            y_min=min(y_min,min(compiled_data[case][i]))
            y_max=max(y_max,max(compiled_data[case][i]))
            plt.plot(compiled_data[case][x_axis_quant], compiled_data[case][i], 
                     compiled_data[case]['plt'][0], 
                     color=compiled_data[case]['plt'][1], 
                     fillstyle=compiled_data[case]['plt'][2], 
                     markersize=mark_sizes[case],
                     label=compiled_data[case]['legend'])
        
        # Adjust x,y limits
        if x_axis_quant=='porosity' or x_axis_quant=='v_avg' or x_axis_quant=='ign':
            x_min=0
        if x_axis_quant=='porosity':
            x_max=100
        elif x_axis_quant=='v_avg':
            x_max=20
        if i=='porosity' or i=='v_avg' or i=='ign':
            y_min=0
        if i=='porosity':
            y_max=100
        elif i=='v_avg':
            y_max=20
        
         #PLot connection (ndBR vs Pe)
#        if x_axis_quant=='Pe' and i=='nd_v_avg':
#            plt.plot([lam_1W[x_axis_quant][0],lam_100W[x_axis_quant][0]], [lam_1W[i][0],lam_100W[i][0]],
#                      '--', color='orange',label='_noLegend_')
#            plt.plot([lam_1W[x_axis_quant][-1],lam_100W[x_axis_quant][-1]], [lam_1W[i][-1],lam_100W[i][-1]],
#                      '--', color='orange',label='_noLegend_')
#            # plt.plot(lam_1W[x_axis_quant], lam_1W[i], '-', color='orange',label='_noLegend_')
#            # plt.plot(lam_100W[x_axis_quant], lam_100W[i], '-', color='orange',label='_noLegend_')
#            plt.plot([base[x_axis_quant][-1],dc_1um[x_axis_quant][-1]], [base[i][-1],dc_1um[i][-1]],
#                     '--', color='blue')
#            plt.plot([base[x_axis_quant][0],mu_3[x_axis_quant][0]], [base[i][0],mu_3[i][0]],
#                     '--', color='green')
        
        if curve_fitting_routine:
            tmd=np.ones(2)
            tmd[0]*=min(base[x_curve][0],dc_1um[x_curve][0],dc_100nm[x_curve][0])#,
               # Cp_Al[x_curve][0],Cp_Cu[x_curve][0],lam_100W[x_curve][0],
               # lam_1W[x_curve][0],mu_3[x_curve][0],mu_4[x_curve][0],
               # Ea_70kJ[x_curve][0],Ea_80kJ[x_curve][0],A0_244e6[x_curve][0],
               # A0_489e5[x_curve][0])
            tmd[1]*=max(base[x_curve][-1],dc_1um[x_curve][-1],dc_100nm[x_curve][-1])#,
               # Cp_Al[x_curve][-1],Cp_Cu[x_curve][-1],lam_100W[x_curve][-1],
               # lam_1W[x_curve][-1],mu_3[x_curve][-1],mu_4[x_curve][-1],
               # Ea_70kJ[x_curve][-1],Ea_80kJ[x_curve][-1],A0_244e6[x_curve][-1],
               # A0_489e5[x_curve][-1])
            if (y_curve in y_logscale) and (x_curve in x_logscale):
                plt.plot(tmd, np.exp(func(np.log(tmd), a_base[0],a_base[1])), '--k')
                plt.plot(tmd, np.exp(func(np.log(tmd), a_dc[0],a_dc[1])), '--b')
                plt.plot(tmd, np.exp(func(np.log(tmd), a_mu[0],a_mu[1])), '--g')
            else:
                plt.plot(tmd, np.exp(func(tmd, a_base[0],a_base[1])), '--k')
                plt.plot(tmd, np.exp(func(tmd, a_dc[0],a_dc[1])), '--b')
                plt.plot(tmd, np.exp(func(tmd, a_mu[0],a_mu[1])), '--g')
        
        plt.xlabel(axis_labels[x_axis_quant])
        plt.ylabel(axis_labels[i])
        if i in y_logscale:
            plt.yscale('log')
            y_min=10**(int(np.log10(y_min))-1)
            y_max=10**(int(np.log10(y_max))+1)
        if x_axis_quant in x_logscale:
            plt.xscale('log')
            x_min=10**(int(np.log10(x_min))-1)
            x_max=10**(int(np.log10(x_max))+1)
        plt.ylim([y_min,y_max])
        plt.xlim([x_min,x_max])
        
#        chartBox = plt.get_position()
#        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
#        ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

        plt.legend(ncol=legend_colns)#, loc="lower right")#, bbox_to_anchor=(0.5, -0.5))
        
#        plt.subplots_adjust(top=0.8)
        plt.tight_layout()
    #    plt.title(titles[i]+' versus '+titles[x_axis_quant])
#        plt.show()
        fig.savefig('Num_'+file_name[i]+'_'+file_name[x_axis_quant]+figType)#,dpi=300)
        plt.close(fig)
    
########################## ALL cases
if all_cases:
    # Transport graphs
    for i in y_axis_quant:
#        x_min,x_max,y_min,y_max=10**6,0,10**6,0
        # Get axis limits (non-log scales)
        if x_axis_quant=='porosity' or x_axis_quant=='v_avg' or x_axis_quant=='ign':
            x_min=0
        else:
            x_min=min(min(base[x_axis_quant]),min(dc_1um[x_axis_quant]),min(dc_100nm[x_axis_quant]),
                    min(Cp_Al[x_axis_quant]),min(Cp_Cu[x_axis_quant]),min(lam_100W[x_axis_quant]),
                    min(lam_1W[x_axis_quant]),min(mu_3[x_axis_quant]),min(mu_4[x_axis_quant]),
                    min(Ea_70kJ[x_axis_quant]),min(Ea_80kJ[x_axis_quant]),min(A0_244e6[x_axis_quant]),
                    min(A0_489e5[x_axis_quant]))
        if x_axis_quant=='porosity':
            x_max=100
        elif x_axis_quant=='v_avg':
            x_max=20
        elif x_axis_quant=='ign':
            x_max=2
        else:
            x_max=max(max(base[x_axis_quant]),max(dc_1um[x_axis_quant]),max(dc_100nm[x_axis_quant]),
                    max(Cp_Al[x_axis_quant]),max(Cp_Cu[x_axis_quant]),max(lam_100W[x_axis_quant]),
                    max(lam_1W[x_axis_quant]),max(mu_3[x_axis_quant]),max(mu_4[x_axis_quant]),
                    max(Ea_70kJ[x_axis_quant]),max(Ea_80kJ[x_axis_quant]),max(A0_244e6[x_axis_quant]),
                    max(A0_489e5[x_axis_quant]))
        
        if i=='porosity' or i=='v_avg' or i=='ign':
            y_min=0
        else:
            y_min=min(min(base[i]),min(dc_1um[i]),min(dc_100nm[i]),
                    min(Cp_Al[i]),min(Cp_Cu[i]),min(lam_100W[i]),
                    min(lam_1W[i]),min(mu_3[i]),min(mu_4[i]),
                    min(Ea_70kJ[i]),min(Ea_80kJ[i]),min(A0_244e6[i]),
                    min(A0_489e5[i]))
        if i=='porosity':
            y_max=100
        elif i=='v_avg':
            y_max=20
        elif i=='ign':
            y_max=2
        else:
            y_max=max(max(base[i]),max(dc_1um[i]),max(dc_100nm[i]),
                    max(Cp_Al[i]),max(Cp_Cu[i]),max(lam_100W[i]),
                    max(lam_1W[i]),max(mu_3[i]),max(mu_4[i]),
                    max(Ea_70kJ[i]),max(Ea_80kJ[i]),max(A0_244e6[i]),
                    max(A0_489e5[i]))
        
        # Plot transport cases (kinetics and transport separate)
#        fig=plt.figure(figsize=fig_size)
#        for case in data_keys_trans:
#            if case=='Cp_Al' and (i=='ign' or x_axis_quant=='ign'):
#                continue
#            elif case=='Cp_Cu' and (i=='ign' or x_axis_quant=='ign'):
#                continue
#            else:
#                plt.plot(compiled_data[case][x_axis_quant], compiled_data[case][i], 
#                         compiled_data[case]['plt'][0], 
#                         color=compiled_data[case]['plt'][1], 
#                         fillstyle=compiled_data[case]['plt'][2], 
#                         label=compiled_data[case]['legend'])
#        
#        # Plot connecting lines between data points
#        if x_axis_quant=='Pe' and i=='nd_v_avg':
#            plt.plot([lam_1W[x_axis_quant][0],lam_100W[x_axis_quant][0]], [lam_1W[i][0],lam_100W[i][0]],
#                      '--', color='orange')
#            plt.plot(lam_1W[x_axis_quant], lam_1W[i], '-', color='orange')
#            plt.plot(lam_100W[x_axis_quant], lam_100W[i], '-', color='orange')
#    #        plt.plot([base[x_axis_quant][-1],dc_1um[x_axis_quant][-1]], [base[i][-1],dc_1um[i][-1]],
#    #                  '--', color='blue')
#    #        plt.plot([base[x_axis_quant][0],mu_3[x_axis_quant][0]], [base[i][0],mu_3[i][0]],
#    #                  '--', color='green')
#        if curve_fitting_routine:
#            tmd=np.ones(2)
#            tmd[0]*=min(base[x_curve][0],dc_1um[x_curve][0],dc_100nm[x_curve][0])#,
#               # Cp_Al[x_curve][0],Cp_Cu[x_curve][0],lam_100W[x_curve][0],
#               # lam_1W[x_curve][0],mu_3[x_curve][0],mu_4[x_curve][0],
#               # Ea_70kJ[x_curve][0],Ea_80kJ[x_curve][0],A0_244e6[x_curve][0],
#               # A0_489e5[x_curve][0])
#            tmd[1]*=max(base[x_curve][-1],dc_1um[x_curve][-1],dc_100nm[x_curve][-1])#,
#               # Cp_Al[x_curve][-1],Cp_Cu[x_curve][-1],lam_100W[x_curve][-1],
#               # lam_1W[x_curve][-1],mu_3[x_curve][-1],mu_4[x_curve][-1],
#               # Ea_70kJ[x_curve][-1],Ea_80kJ[x_curve][-1],A0_244e6[x_curve][-1],
#               # A0_489e5[x_curve][-1])
#            if (y_curve in y_logscale) and (x_curve in x_logscale):
#                plt.plot(tmd, np.exp(func(np.log(tmd), a_base[0],a_base[1])), '--k')
#                plt.plot(tmd, np.exp(func(np.log(tmd), a_dc[0],a_dc[1])), '--b')
#                plt.plot(tmd, np.exp(func(np.log(tmd), a_mu[0],a_mu[1])), '--g')
#            else:
#                plt.plot(tmd, np.exp(func(tmd, a_base[0],a_base[1])), '--k')
#                plt.plot(tmd, np.exp(func(tmd, a_dc[0],a_dc[1])), '--b')
#                plt.plot(tmd, np.exp(func(tmd, a_mu[0],a_mu[1])), '--g')
#        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#        
##        length=np.sqrt((lam_100W[x_axis_quant][0]-lam_1W[x_axis_quant][0])**2+\
##                  (lam_100W[i][0]-lam_1W[i][0])**2)
##        plt.arrow(lam_1W[x_axis_quant][0], lam_1W[i][0],
##                  lam_100W[x_axis_quant][0]-lam_1W[x_axis_quant][0],
##                  lam_100W[i][0]-lam_1W[i][0],
##                  length_includes_head=True,
##                  width=10**(-11), 
##                  head_width=0.000001*length, 
##                  head_length=0.01*length,
##                  color='orange')
#        plt.xlabel(axis_labels[x_axis_quant])
#        plt.ylabel(axis_labels[i])
#        plt.tight_layout()
#        if i in y_logscale:
#            plt.yscale('log')
#            y_min=10**(int(np.log10(y_min))-1)
#            y_max=10**(int(np.log10(y_max))+1)
#        if x_axis_quant in x_logscale:
#            plt.xscale('log')
#            x_min=10**(int(np.log10(x_min))-1)
#            x_max=10**(int(np.log10(x_max))+1)
#        plt.ylim([y_min,y_max])
#        plt.xlim([x_min,x_max])
#        plt.legend(ncol=legend_colns)
#    #    plt.title(titles[i]+' versus '+titles[x_axis_quant])
##        fig.savefig('Num_'+file_name[i]+'_'+file_name[x_axis_quant]+'_ALL'+figType)#,dpi=300)
#        fig.savefig('Num_'+file_name[i]+'_'+file_name[x_axis_quant]+'_trans'+figType)#,dpi=300)
#        plt.close(fig)
#        
#        # Kinetics phenomena
#        fig=plt.figure(figsize=fig_size)
#        for case in data_keys_kin:
#            if case=='A0_489e5' and (i=='ign' or x_axis_quant=='ign'):
#                continue
#            else:
#                plt.plot(compiled_data[case][x_axis_quant], compiled_data[case][i], 
#                         compiled_data[case]['plt'][0], 
#                         color=compiled_data[case]['plt'][1], 
#                         fillstyle=compiled_data[case]['plt'][2], 
#                         label=compiled_data[case]['legend'])
#
##        if i!='ign' and x_axis_quant!='ign':
##            plt.plot(A0_489e5[x_axis_quant], A0_489e5[i], markers['A0'], color='cyan', fillstyle=alph_low, label=A0_489e5['legend'])
#        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#        plt.xlabel(axis_labels[x_axis_quant])
#        plt.ylabel(axis_labels[i])
#        plt.tight_layout()
#        if i in y_logscale:
#            plt.yscale('log')
#        if x_axis_quant in x_logscale:
#            plt.xscale('log')
#        plt.ylim([y_min,y_max])
#        plt.xlim([x_min,x_max])
#        plt.legend()
#    #    plt.title(titles[i]+' versus '+titles[x_axis_quant])
#        fig.savefig('Num_'+file_name[i]+'_'+file_name[x_axis_quant]+'_kin'+figType)#,dpi=300)
#        plt.close(fig)
        
        # Marker size adjustments (manual for each plot)
#        mark_sizes={'base': 6,'dc_1um': 6,'dc_100nm':6,
#                    'Cp_Al': 6,'Cp_Cu': 6,
#                    'lam_100W': 4,'lam_1W': 6,
#                    'mu_3': 6,'mu_4': 6,
#                    'A0_244e6': 6,'A0_489e5': 5,
#                    'Ea_80kJ': 7,'Ea_70kJ': 6
#                } # Pe vs TMD
#        mark_sizes={'base': 6,'dc_1um': 5,'dc_100nm':6,
#                    'Cp_Al': 6,'Cp_Cu': 6,
#                    'lam_100W': 4,'lam_1W': 6,
#                    'mu_3': 6,'mu_4': 6,
#                    'A0_244e6': 6,'A0_489e5': 5,
#                    'Ea_80kJ': 7,'Ea_70kJ': 6
#                } # Pe vs ndBR
#        mark_sizes={'base': 6,'dc_1um': 4,'dc_100nm':7,
#                    'Cp_Al': 6,'Cp_Cu': 6,
#                    'lam_100W': 4,'lam_1W': 4,
#                    'mu_3': 3,'mu_4': 7,
#                    'A0_244e6': 6,'A0_489e5': 6,
#                    'Ea_80kJ': 6,'Ea_70kJ': 6
#                }# ign vs TMD
        mark_sizes={'base': 6,'dc_1um': 6,'dc_100nm':6,
                    'Cp_Al': 6,'Cp_Cu': 6,
                    'lam_100W': 6,'lam_1W': 6,
                    'mu_3': 6,'mu_4': 6,
                    'A0_244e6': 6,'A0_489e5': 6,
                    'Ea_80kJ': 6,'Ea_70kJ': 6
                }
        # Use plt.subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=fig_size)
        leg_entries=[]
        leg_labels=[]
        # Plot transport quantities
        for case in data_keys_trans:
            if case=='Cp_Al' and (i=='ign' or i=='Da' or i=='nd_v_avg'):
                continue
            elif case=='Cp_Cu' and (i=='ign' or i=='Da' or i=='nd_v_avg'):
                continue
            else:
                leg_entries.append(ax1.plot(compiled_data[case][x_axis_quant], 
                                            compiled_data[case][i], 
                                            compiled_data[case]['plt'][0], 
                                            color=compiled_data[case]['plt'][1], 
                                            fillstyle=compiled_data[case]['plt'][2],
                                            markersize=mark_sizes[case],
                                            label=compiled_data[case]['legend'])[0])
                leg_labels.append(compiled_data[case]['legend'])
        
        # Plot kinetic quantities    
        ax2.plot(compiled_data['base'][x_axis_quant], 
                             compiled_data['base'][i], 
                             compiled_data['base']['plt'][0], 
                             color=compiled_data['base']['plt'][1], 
                             fillstyle=compiled_data['base']['plt'][2],
                             label='_noLegend_')
        for case in data_keys_kin:
            if case=='A0_489e5' and (i=='ign' or x_axis_quant=='ign'):
                continue
            else:
                leg_entries.append(ax2.plot(compiled_data[case][x_axis_quant], 
                                            compiled_data[case][i], compiled_data[case]['plt'][0], 
                                            color=compiled_data[case]['plt'][1], 
                                            fillstyle=compiled_data[case]['plt'][2], 
                                            markersize=mark_sizes[case],
                                            label=compiled_data[case]['legend'])[0])
                leg_labels.append(compiled_data[case]['legend'])
        
        if i in y_logscale:
            plt.yscale('log')
            y_min=10**(int(np.log10(y_min))-1)
            y_max=10**(int(np.log10(y_max))+1)
        if x_axis_quant in x_logscale:
            plt.xscale('log')
            x_min=10**(int(np.log10(x_min))-1)
            x_max=10**(int(np.log10(x_max))+1)
        ax1.set_xlabel(axis_labels[x_axis_quant])
        ax2.set_xlabel(axis_labels[x_axis_quant])
        ax1.set_ylabel(axis_labels[i])
        ax1.annotate('(a)',(85,1.6)) # (85,10**(-3)) (85,16) (10**(-3),10**8)
        ax2.annotate('(b)',(85,1.6)) # (85,10**(-3)) (85,16) (10**(-3),10**8)
        plt.ylim([y_min,y_max])
        plt.xlim([x_min,x_max])
        fig.legend(loc=8, ncol=4)#leg_entries, labels=leg_labels, )
        if width==460:
#            if (i=='ign' or i=='Da'):
#                plt.subplots_adjust(bottom=0.3)
#            elif (i=='ign' or i=='Da'):
#                plt.subplots_adjust(bottom=0.3)
#            else:
            plt.subplots_adjust(bottom=0.35)
        else:
            plt.subplots_adjust(bottom=0.4)
        
        # Plot connecting lines between data points
#        if x_axis_quant=='Pe' and i=='nd_v_avg':
#            ax1.plot([lam_1W[x_axis_quant][0],lam_100W[x_axis_quant][0]], [lam_1W[i][0],lam_100W[i][0]],
#                      '--', color='orange',label='_noLegend_')
#            ax1.plot([lam_1W[x_axis_quant][-1],lam_100W[x_axis_quant][-1]], [lam_1W[i][-1],lam_100W[i][-1]],
#                      '--', color='orange',label='_noLegend_')
#            # ax1.plot(lam_1W[x_axis_quant], lam_1W[i], '-', color='orange',label='_noLegend_')
#            # ax1.plot(lam_100W[x_axis_quant], lam_100W[i], '-', color='orange',label='_noLegend_')
#            ax1.plot([base[x_axis_quant][-1],dc_1um[x_axis_quant][-1]], [base[i][-1],dc_1um[i][-1]],
#                     '--', color='blue')
#            ax1.plot([base[x_axis_quant][0],mu_3[x_axis_quant][0]], [base[i][0],mu_3[i][0]],
#                     '--', color='green')
#        plt.tight_layout()
        
        if curve_fitting_routine:
            tmd=np.ones(2)
            tmd[0]*=min(base[x_curve][0],dc_1um[x_curve][0],dc_100nm[x_curve][0],
               # Cp_Al[x_curve][0],Cp_Cu[x_curve][0],lam_100W[x_curve][0],
               # lam_1W[x_curve][0],
               mu_3[x_curve][0],mu_4[x_curve][0],
               # Ea_70kJ[x_curve][0],Ea_80kJ[x_curve][0],A0_244e6[x_curve][0],
               # A0_489e5[x_curve][0]
               )
            tmd[1]*=max(base[x_curve][-1],dc_1um[x_curve][-1],dc_100nm[x_curve][-1],
               # Cp_Al[x_curve][-1],Cp_Cu[x_curve][-1],lam_100W[x_curve][-1],
               # lam_1W[x_curve][-1],
               mu_3[x_curve][-1],mu_4[x_curve][-1],
               # Ea_70kJ[x_curve][-1],Ea_80kJ[x_curve][-1],A0_244e6[x_curve][-1],
               # A0_489e5[x_curve][-1]
               )
            if (y_curve in y_logscale) and (x_curve in x_logscale):
                ax1.plot(tmd, np.exp(func(np.log(tmd), a_base[0],a_base[1])), '--k')
                ax2.plot(tmd, np.exp(func(np.log(tmd), a_base[0],a_base[1])), '--k')
                ax1.plot(tmd, np.exp(func(np.log(tmd), a_dc[0],a_dc[1])), '--b')
                ax1.plot(tmd, np.exp(func(np.log(tmd), a_mu[0],a_mu[1])), '--g')
            else:
                ax1.plot(tmd, np.exp(func(tmd, a_base[0],a_base[1])), '--k')
                ax2.plot(tmd, np.exp(func(tmd, a_base[0],a_base[1])), '--k')
                ax1.plot(tmd, np.exp(func(tmd, a_dc[0],a_dc[1])), '--b')
                ax1.plot(tmd, np.exp(func(tmd, a_mu[0],a_mu[1])), '--g')
#        plt.show()
        fig.savefig('Num_'+file_name[i]+'_'+file_name[x_axis_quant]+'_ALL'+figType)#,dpi=300)
        plt.close(fig)
        
########################## mesh study
if mesh_study:
    fig=plt.figure(figsize=fig_size)
    plt.plot(mesh['porosity'], mesh[mesh_quant])
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel('Grid Points')
    plt.xscale('log')
    plt.ylabel(axis_labels[mesh_quant])
#    plt.legend()
    #plt.title(titles[mesh_quant]+' versus mesh')
    fig.savefig('Num_mesh'+figType)#,dpi=300)
    plt.close(fig)

########################## Compare with experiments
# These dictionaries contain data from the 'Summary.xlsm' spreadsheet; worksheet called 'Graphs'
Sacel_exp={
        'Pe': [2.63724E-07,2.57947E-08,2.57947E-08,2.31367E-08,1.49063E-09,1.20325E-09,1.20325E-09,2.67549E-07,9.33094E-09,7.63223E-09,2.82675E-09,1.24512E-09,2.3445E-07,1.00207E-08,2.47705E-09,1.83108E-09,9.80872E-10], # R as non-dim
#        'Pe': [1.2557E-06,1.2282E-07,1.2282E-07,1.1017E-07,7.0977E-09,5.7293E-09,5.7293E-09,1.2739E-06,4.4430E-08,3.6341E-08,1.3460E-08,5.9287E-09,1.1163E-06,4.7714E-08,1.1795E-08,8.7187E-09,4.6705E-09], # Cp as non-dim
        'nd_v_avg': [856232.9981,379733.3172,284799.9879,529195.8652,3285552.673,3052699.87,7122966.364,4699874.171,2192110.524,4020016.29,7236029.322,16427763.36,4699874.171,1788691.33,2532610.263,19577556.39,27410315.51]
        }
Weis_exp={
        'Pe': [0.000123441,0.000277742,0.000493763,0.001110968,0.002499677,0.003086022,0.00999871,0.012344086,0.022497097],
#        'Pe': [8.4874E-03,1.9097E-02,3.3950E-02,7.6386E-02,1.7187E-01,2.1218E-01,6.8748E-01,8.4874E-01,1.5468E+00], # Cp as non dim
        'nd_v_avg': [2.74E+05,2.06E+05,1.20E+05,3.43E+04,3.05E+03,2.06E+03,5.71E+02,6.85E+01,3.81E+01]
        }
Granier_exp={
        'Pe': [1.362E-07,1.146E-07,9.810E-08,8.449E-08,7.326E-08,8.139E-08,7.001E-08,6.029E-08,5.168E-08,4.460E-08],
#        'Pe': [6.4844E-07,5.4572E-07,4.6710E-07,4.0232E-07,3.4885E-07,3.8755E-07,3.3337E-07,2.8707E-07,2.4606E-07,2.1235E-07], # Cp as non-dime
        'nd_v_avg': [1.211E+04,6.054E+04,9.687E+04,2.422E+04,1.211E+04,4.305E+04,8.611E+04,1.292E+05,2.153E+04,2.153E+04]
        }
Ahn_exp={
        'Pe': [1.332E-07,6.629E-08,3.300E-08,8.188E-09,6.998E-10],
#        'Pe': [6.3410E-07,3.1565E-07,1.5713E-07,3.8988E-08,3.3323E-09], # Cp as non-dim
        'nd_v_avg': [1.757E+06,2.715E+06,5.000E+06,1.832E+07,1.929E+08]
        }
Dean_exp={
        'Pe': [2.783E-07,2.353E-07,2.037E-07,1.919E-07,1.882E-07,1.847E-07,1.813E-07,1.704E-07,1.594E-07],
#        'Pe': [1.3253E-06,1.1202E-06,9.7004E-07,9.1354E-07,8.9614E-07,8.7939E-07,8.6325E-07,8.1116E-07,7.5883E-07], # Cp as non-dim
        'nd_v_avg': [3.245E+06,3.245E+06,3.353E+06,3.245E+06,2.704E+06,4.219E+06,3.245E+06,2.704E+06,2.434E+06]
        }

if exp_comp:
    x_axis_quant='Pe'
    i='nd_v_avg'
    fig=plt.figure(figsize=fig_size)
    plt.plot(base[x_axis_quant], base[i], markers['mu'], color='black', label='Numerical')
    plt.plot(dc_1um[x_axis_quant], dc_1um[i], markers['mu'], color='black')
    plt.plot(dc_100nm[x_axis_quant], dc_100nm[i], markers['mu'], color='black')
    plt.plot(Cp_Cu[x_axis_quant], Cp_Cu[i], markers['mu'], color='black')
    plt.plot(Cp_Al[x_axis_quant], Cp_Al[i], markers['mu'], color='black')
    plt.plot(lam_100W[x_axis_quant], lam_100W[i], markers['mu'], color='black')
    plt.plot(lam_1W[x_axis_quant], lam_1W[i], markers['mu'], color='black')
    plt.plot(mu_4[x_axis_quant], mu_4[i], markers['mu'], color='black')
    plt.plot(mu_3[x_axis_quant], mu_3[i], markers['mu'], color='black')
    plt.plot(Ea_70kJ[x_axis_quant], Ea_70kJ[i], markers['mu'], color='black')
    plt.plot(Ea_80kJ[x_axis_quant], Ea_80kJ[i], markers['mu'], color='black')
    plt.plot(A0_244e6[x_axis_quant], A0_244e6[i], markers['mu'], color='black')
    plt.plot(A0_489e5[x_axis_quant], A0_489e5[i], markers['mu'], color='black')
    plt.plot(Sacel_exp[x_axis_quant], Sacel_exp[i], markers['base'], label='Saceleanu')
    plt.plot(Weis_exp[x_axis_quant], Weis_exp[i], markers['A0'], label='Weismiller')
    #plt.plot(Granier_exp[x_axis_quant], Granier_exp[i], markers['dc'], color='black', label='Granier')
    plt.plot(Ahn_exp[x_axis_quant], Ahn_exp[i], markers['lam'], label='Ahn')
    #plt.plot(Dean_exp[x_axis_quant], Dean_exp[i], markers['Cp'], color='black', label='Dean')
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel(axis_labels[x_axis_quant])
    plt.ylabel(axis_labels[i])
    plt.tight_layout()
    if i in y_logscale:
        plt.yscale('log')
    if x_axis_quant in x_logscale:
        plt.xscale('log')
    plt.legend()
    #plt.title(titles[i]+' versus '+titles[x_axis_quant])
    fig.savefig('Num_exp_'+file_name[i]+'_'+file_name[x_axis_quant]+figType)#,dpi=300)
    plt.close(fig)

########################## Thermal conductivity models
if therm_cond_models:
    porosity=np.linspace(0,1,11)
    ks,kf=100,0.01
    alpha=kf/ks
    k_geom=ks*(kf/ks)**porosity
    k_parallel=ks*(1-porosity)+kf*porosity
    k_series=((1-porosity)/ks+porosity/kf)**(-1)
    k_maxwell=ks+ks*((3*alpha-3)*porosity)/(alpha+2-(alpha-1)*porosity)
    #k_EMT=emt(ks,kf,porosity)
    k_EMT=[100,85.0024,70.0051,55.0086,40.0135,25.0225,10.0537,0.09818,0.0249775,0.01428335,0.01]
    k_recip=ks*(1+(np.sqrt(alpha)-1)*porosity)/(1+(np.sqrt(1/alpha)-1)*porosity)
    k_stacy0=np.ones_like(porosity)*0.2
    k_stacy1=np.ones_like(porosity)
    
    fig=plt.figure(figsize=fig_size)
    plt.plot(porosity, k_parallel, markers['Cp'], color='blue', label='Parallel')
    plt.plot(porosity, k_maxwell, markers['mu'], color='orange', label='Maxwell')
    plt.plot(porosity, k_EMT, markers['dc'], color='cyan', label='EMT')
    plt.plot(porosity, k_geom, markers['base'], color='magenta', label='Geometric')
    plt.plot(porosity, k_recip, markers['A0'], color='green', label='Reciprocity')
    plt.plot(porosity, k_series, markers['lam'], color='red', label='Series')
#    plt.plot(porosity, k_stacy0, '--k', label='Stacy')
    plt.plot(porosity, k_stacy1, '--k', label='Stacy')
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel('Porosity')
    plt.ylabel('Effective Conductivity [$Wm^{-1}K^{-1}$]')
    if therm_cond_model_zoom:
        plt.ylim([0,10])
    plt.xlim([0,1])
    #    plt.title(titles[i]+' versus '+titles[x_axis_quant])
    if therm_cond_model_zoom:
        fig.savefig('Models_effective_conductivity_zoom'+figType)#,dpi=300)
    else:
        plt.legend()
        fig.savefig('Models_effective_conductivity'+figType)#,dpi=300)
    plt.close(fig)

########################## Inst BR progression with time
if inst_BR_time:
    # Curve fit
    t=np.ones(4)*[BR_t1[0][0],BR_t2[0][1],BR_t3[0][2],BR_t4[0][4]]
    v=np.ones(4)*[BR_t1[1][0],BR_t2[1][1],BR_t3[1][2],BR_t4[1][4]]
    a,R=curve_fit_res(t,v, 'quadratic')
    print('Residual for inst_BR curve: '+str(R))
    t_est=np.linspace(0.05,t[-1],10)
    v_est=func_quad(t_est, a[0], a[1], a[2])
    # Plot
    fig=plt.figure(figsize=fig_size)
    plt.plot(BR_t0[0], BR_t0[1], markers['base'], color='black', label=BR_t0[2])
    plt.plot(BR_t1[0], BR_t1[1], markers['Cp'], color='black', label=BR_t1[2])
    plt.plot(BR_t2[0], BR_t2[1], markers['dc'], color='black', label=BR_t2[2])
    plt.plot(BR_t3[0], BR_t3[1], markers['A0'], color='black', label=BR_t3[2])
    plt.plot(BR_t4[0], BR_t4[1], markers['mu2'], color='black', label=BR_t4[2])
    plt.plot(t_est, v_est, '--k')
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel('Time [-]')
    plt.ylabel('$V_{cw}$ [$ms^{-1}$]')
    plt.xlim([0,1])
    plt.legend(loc=(0.4,0.5))
    fig.savefig('Num_BRinst_time'+figType)#,dpi=300)
#    plt.show()
    plt.close(fig)

########################## Darcy contours
# Darcy velocities and pressure contours
if is_v_contours:
    # 90% TMD
    if is_larger_figs:
        os.chdir('..\\..\\1_base\\1')
    else:
        os.chdir('..\\1_base\\1')
    P_90=np.load('P_1.300010.npy')
#    P_90=np.load('P_1.200012.npy')
    X=np.load('X.npy', False)
    Y=np.load('Y.npy', False)
    v_90=np.zeros_like(P_90)
    por=np.ones_like(P_90)*0.1
    perm=por**3*(40e-9)**2/(72*(1-por)**2)
    v_90[1:,:]=-interpolate(perm[1:,:], perm[:-1,:], 'harmonic')\
        /(10**(-5))*(P_90[1:,:]-P_90[:-1,:])/(Y[1:,:]-Y[:-1,:])
    
    # 10% TMD
    os.chdir('..\\5')
    P_10=np.load('P_0.275000.npy')
    v_10=np.zeros_like(P_10)
    por=np.ones_like(P_10)*0.9
    perm=por**3*(40e-9)**2/(72*(1-por)**2)
    v_10[1:,:]=-interpolate(perm[1:,:], perm[:-1,:], 'harmonic')\
        /(10**(-5))*(P_10[1:,:]-P_10[:-1,:])/(Y[1:,:]-Y[:-1,:])
    
    if is_larger_figs:
        os.chdir('..\\..\\'+folder_output+'\\Larger')
    else:
        os.chdir('..\\..\\'+folder_output)
    # Darcy Velocity contours
    lvl_v=np.linspace(-24, 24, 7)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=fig_size)
    contour_90=ax1.contourf(X*1000, Y*1000, v_90, alpha=0.5, extend='both',levels=lvl_v)#, vmin=270, vmax=2000)  
    ax1.contour(contour_90, colors='k')
    contour_10=obj=ax2.contourf(X*1000, Y*1000, v_10, alpha=0.5, extend='both',levels=lvl_v)#, vmin=270, vmax=2000)  
    ax2.contour(contour_10, colors='k')
    ax1.set_xlabel('$r$ [mm]')
    ax2.set_xlabel('$r$ [mm]')
    ax1.set_ylabel('$z$ [mm]')
    ax1.annotate('(a)',(0.85,3.65))
    ax2.annotate('(b)',(0.85,3.65))
    plt.ylim([2.55,3.8])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(obj, cax=cbar_ax)
    fig.savefig('Num_Vcontour_ALL'+figType)
#    plt.show()
    plt.close(fig)

########################## Trans pressure and temperature
if is_P_trans:
    if is_larger_figs:
        os.chdir('..\\..\\1_base\\5')
    else:
        os.chdir('..\\1_base\\5')
    times=os.listdir('.')
    i=len(times)
    j=0
    while i>j:
        if st.find(times[j],'T')==0 and st.find(times[j],'.npy')>0:
            times[j]=st.split(st.split(times[j],'_')[1],'.npy')[0]
            j+=1
        else:
            del times[j]
            i-=1
    
    Phi_graphs=(1117,112)
    Temps=[]
    Press=[]
    t=[]
    times.sort()
    for time in times:
        t.append(float(time))
        
        T=np.load('P_'+time+'.npy', False)
        Press.append(T[Phi_graphs])
        
        T=np.load('T_'+time+'.npy', False)
        Temps.append(T[Phi_graphs])
        
    fig,ax1=plt.subplots()
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('P [$Pa$]', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('T [$K$]', color='red')  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.plot(t, Press, color='black')
    ax2.plot(t, Temps, color='red')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if is_larger_figs:
        os.chdir('..\\..\\'+folder_output+'\\Larger')
    else:
        os.chdir('..\\..\\'+folder_output)
    fig.savefig('Num_P_T_10TMD'+figType)
    plt.close(fig)
    
    
print '\nPost-processing complete'