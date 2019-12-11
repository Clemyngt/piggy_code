# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:27:10 2019

@author: Binet Photo
"""

import pykep as pk
import pygmo as pg
import numpy as np
from matplotlib import pyplot as plt
from pykep.examples import add_gradient, algo_factory

algo = algo_factory("slsqp")

udp = add_gradient(pk.trajopt.direct_pl2pl(
        p0="earth",
        pf="mars",
        mass=300,
        thrust=0.1,
        isp=3000,
        vinf_arr=1e-6,
        vinf_dep=3.5,
        hf=False,
        nseg=40,
        t0=[7300, 9125],
        tof=[100, 500]),
        with_grad=True
    )

prob = pg.problem(udp)

prob.c_tol = [1e-5] * prob.get_nc()

pop = pg.population(prob, 1)

pop = algo.evolve(pop)

if prob.feasibility_x(pop.champion_x):
    print("Optimal Found!!")
else:
    print("No solution found, try again :)")

udp.udp_inner.pretty(pop.champion_x)

axis = udp.udp_inner.plot_traj(pop.champion_x)
plt.title("The trajectory in the heliocentric frame")
axis = udp.udp_inner.plot_control(pop.champion_x)
plt.title("The control profile (throttle)")

plt.ion()
plt.show()