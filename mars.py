#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:24:41 2019

@author: a
"""
# Imports
import pykep as pk
import pygmo as pg
import numpy as np

# Plotting imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

DV_max = 200
N_max = 6000 // DV_max

# We define the optimization problem
udp = pk.trajopt.pl2pl_N_impulses(
    start=pk.planet.jpl_lp('earth'),
    target=pk.planet.jpl_lp('mars'),
    N_max=N_max,
    tof=[100., 1000.],
    vinf=[0., float(DV_max) / 1000],
    phase_free=False,
    multi_objective=False,
    t0=[pk.epoch(0), pk.epoch(1000)])
# All pykep problems in the module trajopt are compatible with pygmo.
# So we create a pygmo problem from the pykep udp (User Defined Problem)
prob = pg.problem(udp)
print(prob)

# Here we define the solution strategy, which in this simple case is to use
# Covariance Matrix adaptation Evolutionary Strategy (CMA-ES)
uda = pg.cmaes(gen=1000, force_bounds = True)
algo = pg.algorithm(uda)
# Here we activate some degree of screen output (will only show in the terminal)
algo.set_verbosity(10)
# We construct a random population of 20 individuals (the initial guess)
pop = pg.population(prob, size = 20, seed = 123)
# We solve the problem
pop = algo.evolve(pop)

# Plot our trajectory
fig = plt.figure(figsize = (16,5))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax1 = udp.plot(pop.champion_x, axes = ax1)
ax2 = udp.plot(pop.champion_x, axes = ax2)
ax2.view_init(elev=90, azim=0)
ax3 = udp.plot(pop.champion_x, axes = ax3)
ax3.view_init(elev=0, azim=0)

udp.pretty(pop.champion_x)

print(DV_max, N_max)
plt.show()

## 3
#print(sum([939.5724879571543, 1980.2346728149923, 2712.4971488556344]))

## 10 
#print(sum([2.324166997839415e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1880233060147024e-05, 0.0, 2969.4684168139634, 2700.2260194993346]))

## 15
#print(sum([0.0, 0.0, 0.0, 3.0321431626084934, 0.0, 0.12882088451337018, 0.0, 995.679984452365, 0.3903853009126621, 0.0, 0.0, 0.0, 0.0, 2445.8736585190863, 2237.9490964749943]))


