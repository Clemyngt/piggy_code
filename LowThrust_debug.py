# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:27:10 2019

@author: Binet Photo
"""

import pykep as pk
import pygmo as pg
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pykep.examples import add_gradient, algo_factory

algo = algo_factory("slsqp")

udp = add_gradient(pk.trajopt.direct_pl2pl(
        p0="earth",
        pf="earth",
        mass=300,
        thrust=0.1,
        isp=3000,
        vinf_arr=0,#1e-6,
        vinf_dep=0,#3.5,
        hf=False,
        nseg=4,
        t0=[7300, 9125],
        tof=[100, 500]),
        with_grad=True
    )

prob = pg.problem(udp)

prob.c_tol = [1] * prob.get_nc()

pop = pg.population(prob, 1)

pop = algo.evolve(pop)

if prob.feasibility_x(pop.champion_x):
    print("Optimal Found!!")
else:
    print("No solution found, try again :)")

traj = udp.udp_inner.get_traj(pop.champion_x)
fig = plt.figure()
plt.subplot(221)
plt.plot(traj[:,0], traj[:,8:12])
plt.xlabel("time (d)")
plt.legend(["$|U|$", "$U_x$", "$U_y$", "$U_z$"])

plt.subplot(222)
plt.plot(traj[:,0], traj[:,1:4] / 149597870700)
plt.xlabel("time (d)")
plt.ylabel("position (A.U.)")
plt.legend(["x", "y", "z"])

plt.subplot(223)
plt.plot(traj[:,0], traj[:,5:8])
plt.xlabel("time (d)")
plt.ylabel("speed (m/s)")
plt.legend(["$V_x$", "$V_y$", "$V_z$"])

ax = fig.add_subplot(224, projection='3d')
ax.plot(traj[:,1], traj[:,2], traj[:,3])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend(["trajectoire"])

udp.udp_inner.pretty(pop.champion_x)

#axis = udp.udp_inner.plot_traj(pop.champion_x)
#plt.title("The trajectory in the heliocentric frame")
#axis = udp.udp_inner.plot_control(pop.champion_x)
#plt.title("The control profile (throttle)")

#plt.ion()
plt.show()
