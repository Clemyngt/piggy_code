# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 08:25:08 2020

@author: Clément Yang
"""


""" Imports """
# Main imports
from pykep.trajopt import mga_1dsm
from pykep.planet import jpl_lp, keplerian
from pykep import AU, DEG2RAD, MU_SUN, epoch
import pykep as pk
import pygmo as pg
import numpy as np
from pykep.examples import add_gradient, algo_factory

# Plotting imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


""" Define the keplerian body """ 
# with input (date,orbital_elements, mu_central_body, mu_self, radius, safe_radius [, name = ‘unknown’])
# orbital_elements = (a,e,i,W,w,M )
#(we need to modify the safe radius of the planets to match the wanted problem)
asteroid = keplerian(epoch(2458600.5, "jd"), #MJD = JD - 2400000.5
                                 [1.926 * AU,           #a
                                  .474,                 #e
                                  1.68 * DEG2RAD,       #i
                                  285.38 * DEG2RAD,     #W
                                  186.38 * DEG2RAD,     #w
                                  163.82 * DEG2RAD],    #M
                                 MU_SUN,    #mu_central_body
                                 0.,        #mu_self
                                 0.490, 0.55, #radius, safe_radius : safe_radius must be greater than radius
                                 "162998")  #body name

""" Define the UDP (User Defined Problem) """
# Using continuous thrust w/o mga
#udp = add_gradient(pk.trajopt.lt_margo(target = asteroid, 
#                                       n_seg=30, 
#                                       grid_type="uniform",
#                                       t0=[pk.epoch(2459580,'jd'), epoch(2466154,'jd')],  #must be an epoch class
#                                       tof=[1000,2000],
#                                       m0=1000,
#                                       Tmax=0.2,     #using Safran PPS5000 
#                                       Isp=3000,
#                                       earth_gravity=False,  #If True, start cannot be 'earth'
#                                       sep=True,        #sep = True if Solar Electric Propulsion including decreament of power with AU, False if NEP
#                                       start="earth"),  #start can be 'earth', 'L1' or 'L2'
#                                    with_grad=True)     
            
# Using continuous thrust with mga
# udp = add_gradient(pk.trajopt.mga_lt_nep(
#         seq = [pk.planet.jpl_lp('earth'), pk.planet.jpl_lp('venus'), pk.planet.jpl_lp('mercury')],
#         n_seg = [5, 20],
#         t0 = [pk.epoch(2459580,'jd'), epoch(2466154,'jd')], # This is in mjd2000
#         tof = [[100, 1000], [200, 2000]], # This is in days
#         vinf_dep = 3., # This is in km/s
#         vinf_arr = 2., # This is in km/s
#         mass = [1000., 2000.0],
#         Tmax = 0.2,
#         Isp = 3500.0,
#         fb_rel_vel = 6., # This is in km/s
#         multi_objective = False,
#         high_fidelity = False),
#         with_grad=True)

# Using direct method of continuous thrust pl2pl with a change in reference code with attributes p0 and pf
udp = add_gradient(pk.trajopt.direct_pl2pl(
        p0=pk.planet.jpl_lp('earth'),
        pf=asteroid,
        mass=1000,
        thrust=0.5,
        isp=3000,
        vinf_arr=1e-6,      #allowed maximal DV at arriv in [km/s]
        vinf_dep=5,          #allowed maximal DV at departure in [km/s]
        hf=False,           #(``bool``): High-fidelity. Activates a continuous representation for the thrust         
        nseg=40,
        t0=[pk.epoch(2459585,'jd').mjd2000, epoch(2462507,'jd').mjd2000],  #WARNING ! list of floats in mjd2000 
        tof=[500,1000]),
        with_grad=True
    )
prob = pg.problem(udp)
prob.c_tol = 1e-4

""" Define the UDA (User Defined Algorithm) """
### Using slsqp 
uda = pg.nlopt('slsqp')
uda.xtol_rel = 1e-5
uda.ftol_rel = 0
algo = pg.algorithm(uda)
algo.set_verbosity(1)

# We create a population of n_sample random initial guesses
n_sample = 1                #WARNING for direct method, use only one initial sample
pop = pg.population(prob, n_sample)
# And optimize
pop = algo.evolve(pop)

print("Is feasible: ", prob.feasibility_f(pop.champion_f))


















