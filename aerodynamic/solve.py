#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve, fmin, root
from scipy.signal import find_peaks
from matplotlib.patches import Circle
from matplotlib import ticker, cm


G = 6.67e-11
B = 1

V0 = 10e3

planet = "mars"

planets = {
	'earth':{
		'name': 'the Earth',
		'R': 6371e3,
		'M': 5.97e24,
		'rho0': 1.39,
		'hs': 7.9e3,
		'color':'#1f77b4'
	},
	'mars':{
		'name': 'Mars',
		'R': 3389e3,
		'M': 6.39e23,
		#'rho0': 1.39,
		'hs': 7.9e3,
		'rho0': 0.1,
		#'hs': 8e3,
		'color':'#e5a77f'
	},
	'venus':{
		'name': 'Venus',
		'R': 6051e3,
		'M': 4.87e24,
		'rho0': 1.39,
		'hs': 7.9e3,
		'color':'#aaaaaa'
	}
}
	
R = planets[planet]['R']
M = planets[planet]['M']
rho0 = planets[planet]['rho0']
hs = planets[planet]['hs']


def rho(r):
	h = r-R;
	return rho0 * np.exp(-h/hs)

def equations(y, t):
	vr, r, vtheta, theta = y
	
	V = np.sqrt(vr*vr + r*r*vtheta*vtheta)
	drag = 0.5 * rho(r) * V / B
	
	ar = r*vtheta*vtheta - G*M/(r*r) - drag*vr
	atheta = -2*vr*vtheta / r - drag*vtheta
	
	if r < R:
		return [ar, 0, atheta, 0] # on ne bouge plus
	
	dydt = [ar, vr, atheta, vtheta]
	return dydt

def initial(x0, y0, vx0, vy0):
	r0 = np.hypot(x0, y0)
	theta0 = np.arctan2(y0, x0)
	vr0 = (x0*vx0+y0*vy0)/r0
	vtheta0 = (x0*vy0 - y0*vx0)/(r0*r0)
	
	return [vr0, r0, vtheta0, theta0]

def simulate_trajectory(alpha, t_max, n_pts=1000):
	t = np.linspace(0, t_max, n_pts)
	
	if alpha > np.pi/2:
		alpha = np.pi/2
	if alpha < 0:
		alpha = 0
		
	x0 = -3*R
	y0 = 0
	vx0 = V0*np.cos(alpha)
	vy0 = V0*np.sin(alpha)
	
	sol = odeint(equations, initial(x0, y0, vx0, vy0), t)
	
	return sol


def draw_planet(ax):
	ax.add_patch(Circle((0, 0), R, color=planets[planet]['color']))

def draw_atmosphere(ax):
	xmin, xmax = ax.get_xlim()
	ymin, ymax = ax.get_ylim()
	
	vmin = 1e-50
	xx, yy = np.meshgrid(np.linspace(xmin, xmax, 1000), np.linspace(ymin, ymax, 1000))
	rr = np.hypot(xx, yy)
	zz = np.where(rr < R+1e3, 0, rho(rr))
	zz = np.where(zz < vmin, vmin, zz)
	
	im = ax.contourf(xx, yy, zz, locator=ticker.LogLocator(numticks=200), cmap='gray', vmin=vmin)
	cbar = plt.colorbar(im, ax=ax)
	cbar.set_label('atmospheric density ($kg.m^{-3}$)', rotation=270)
	
if __name__ == "__main__":
	
	fig_map, ax_map = plt.subplots()
	fig_plt, ax_plot = plt.subplots(nrows=2, ncols=2)
	
	result = {'alpha':[], 'apogee':[], 'perigee':[], 'T':[]}
	
	for i, t_max in enumerate(np.linspace(10000, 90000, 41)):
		print(t_max)
		def objective(a):
			sol = simulate_trajectory(a, t_max)
			vr, r, vtheta, theta = sol.T
			
			return theta[-1]
		
		res = fmin(objective, [0.5])
		
		alpha = res
		# 2. 
		#alpha = np.pi/2
		sol = simulate_trajectory(alpha, t_max, 20000)
		vr, r, vtheta, theta = sol.T
		
		x = r*np.cos(theta)
		y = r*np.sin(theta)
		
		h = r-R;
		v = np.hypot(vr, vtheta*r)
		#E = 0.5*v*v +G*M / R -  G*M / r
		
		if i%5 == 0:
			ax_map.plot(x, y, '.-', markersize=4, zorder=10, label="T = {}".format(int(t_max)))
		
			ax_plot[0,0].plot(np.linspace(0, t_max, 20000), h*0.001)
		
		apogee = np.max(h[theta < 0])
		perigee = np.min(h[theta > 0])
		result['T'].append(t_max)
		result['alpha'].append(alpha)
		result['apogee'].append(apogee)
		result['perigee'].append(perigee)
		
	draw_atmosphere(ax_map)
	draw_planet(ax_map)
	
	ax_map.set_aspect('equal')
	ax_map.legend()
	print(result)
	
	ax_plot[0,0].set_xlabel("time (s)")
	ax_plot[0,0].set_ylabel("altitude (km)")
	
	ax_plot[0,1].plot(np.array(result['perigee'])*0.001, np.array(result['apogee'])*0.001, 'o-')
	ax_plot[0,1].set_xlabel("perigee (km)")
	ax_plot[0,1].set_ylabel("apogee (km)")
	
	ax_plot[1,1].plot(np.array(result['alpha'])*180/np.pi, np.array(result['apogee'])*0.001, 'o-')
	ax_plot[1,1].set_xlabel("attack angle (°)")
	ax_plot[1,1].set_ylabel("apogee (km)")
	
	ax_plot[1,0].plot(np.array(result['alpha'])*180/np.pi, np.array(result['perigee'])*0.001, 'o-')
	ax_plot[1,0].set_xlabel("attack angle (°)")
	ax_plot[1,0].set_ylabel("perigee (km)")
	
	fig_plt.tight_layout()
	plt.show()
