# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:55:54 2020

@author: Clément Yang
"""

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


# 1 - Algorithm
algo = algo_factory("slsqp")

# 2 - Problem
"""- x0 (``list``, ``tuple``, ``numpy.ndarray``): Departure state [m, m, m, m/s, m/s, m/s, kg].
            - xf (``list``, ``tuple``, ``numpy.ndarray``): Arrival state [m, m, m, m/s, m/s, m/s, kg].
            - tof (``list``): Transfer time bounds [days].
            - thrust (``float``, ``int``): Spacecraft maximum thrust [N].
            - isp (``float``, ``int``): Spacecraft specific impulse [s].
            - mu (``float``): Gravitational parametre of primary body [m^3/s^2].
            - freetime (``bool``): Activates final time transversality condition. Allows final time to vary.
            - alpha (``float``, ``int``): Homotopy parameter (0 -quadratic control, 1 - mass optimal)
            - bound (``bool``): Activates bounded control, in which the control throttle is bounded between 0 and 1, otherwise the control throttle is allowed to unbounded.
            - atol (``float``, ``int``): Absolute integration solution tolerance.
            - rtol (``float``, ``int``): Relative integration solution tolerance.
"""
udp = add_gradient(pk.trajopt.indirect_pt2pt(
    x0=[41724872.10444193, -3820014.9394953824, -0.060797980320508316,
        -191.09331168678096, 1675.2635954692362, -2.821550881333832e-05, 350],
    xf=[6.27655e+09, -9.44995e+09, -3.54473e+08,
        4659.73, -6606.27, -248.972, 300],
    thrust=0.029,
    isp=1700,
    mu=pk.MU_EARTH,
    tof=[50, 100],
    freetime=True,
    alpha=0,    # quadratic control
    bound=True),
    with_grad=True
)
prob = pg.problem(udp)
prob.c_tol = [1e-5] * prob.get_nc()

# 3 - Population
pop = pg.population(prob)
z = np.hstack(([np.random.uniform(udp.udp_inner.tof[0],
                                  udp.udp_inner.tof[1])], 10 * np.random.randn(7)))
pop.push_back(z)

# 4 - Solve the problem (evolve)
pop = algo.evolve(pop)

# 5 - Continue the solution to mass optimal
#homotopy_path = [0.5, 0.75, 0.9, 1]
#for alpha in homotopy_path:
#    z = pop.champion_x
#    print("alpha: ", alpha)
#    udp = add_gradient(pk.trajopt.indirect_pt2pt(
#        x0=[44914296854.488266, -145307873786.94177, 1194292.6437741749,
#            31252.149474878544, 9873.214642584162, -317.08718075574404, 1000],
#        xf=[-30143999066.728119, -218155987244.44385, -3829753551.2279921,
#            24917.707565772216, -1235.74045124602, -638.05209482866155, 905.47894037275546],
#        thrust=0.1,
#        isp=3000,
#        mu=pk.MU_SUN,
#        tof=[616.77087591237546, 616.77087591237546],
#        freetime=False,
#        alpha=alpha,    # quadratic control
#        bound=True),
#        with_grad=True
#    )
#    prob = pg.problem(udp)
#    prob.c_tol = [1e-5] * prob.get_nc()
#
#    # 7 - Solve it
#    pop = pg.population(prob)
#    pop.push_back(z)
#    pop = algo.evolve(pop)

# 8 - Inspect the solution
print("Feasible?:", prob.feasibility_x(pop.champion_x))

# plot trajectory
#axis = udp.udp_inner.plot_traj(pop.champion_x)#, quiver=True, mark="k")
#plt.title("The trajectory in the heliocentric frame")

# plot control
#udp.udp_inner.plot_control(pop.champion_x)
#plt.title("The control profile (throttle)")
#
#plt.ion()
#plt.show()
#
#udp.udp_inner.pretty(pop.champion_x)
#
#print("\nDecision vector: ", list(pop.champion_x))