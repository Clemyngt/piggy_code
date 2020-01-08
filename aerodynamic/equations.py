#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.patches import Circle
from matplotlib import ticker, cm


G = 6.67e-11
B = 1

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
		'rho0': 1.39,
		'hs': 7.9e3,
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
	#print(h, np.exp(-h/hs))
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


def initial(i, N_curves):
	
	x0 = R+650e3
	y0 = R+350e3
	vx0 = 0
	vy0 = np.linspace(-5.18e3, -5.15e3, N_curves)[i]
	
	r0 = np.hypot(x0, y0)
	theta0 = np.arctan2(y0, x0)
	vr0 = (x0*vx0+y0*vy0)/r0
	vtheta0 = (x0*vy0 - y0*vx0)/(r0*r0)
	
	return [vr0, r0, vtheta0, theta0]


	
if __name__ == "__main__":
	from scipy.integrate import odeint
	
	N_curves = 10;
	N_time = 1000;
	
	
	t = np.linspace(0, 20000, N_time)
	
	x, y, h, v = np.zeros((4, N_time, N_curves))
	
	for i in range(N_curves):
		sol = odeint(equations, initial(i, N_curves), t)
	
		vr, r, vtheta, theta = sol.T
		
		x[:,i] = r*np.cos(theta)
		y[:,i] = r*np.sin(theta)
		
		h[:,i] = r-R;
		v[:,i] = np.hypot(vr, vtheta*r)
	
	# plot limits
	border = 400e3
	xlim = [np.min(x)-border, np.max(x)+border]
	ylim = [np.min(y)-border, np.max(y)+border]
	
	# the atmosphere
	vmin = 1e-50
	xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 1000), np.linspace(ylim[0], ylim[1], 1000))
	rr = np.hypot(xx, yy)
	zz = np.where(rr < R+1e3, 0, rho(rr))
	zz = np.where(zz < vmin, vmin, zz)
	
	plt.contourf(xx, yy, zz, locator=ticker.LogLocator(numticks=200), cmap='gray', vmin=vmin)
	cbar = plt.colorbar()
	cbar.set_label('atmospheric density ($kg.m^{-3}$)', rotation=270)
	
	# draw the trajectories
	plt.plot(x, y, '.-', markersize=4, alpha=0.5)
	#plt.plot(t, r, 'g', label='r(t)')
	#plt.legend()
	plt.xlim(xlim)
	plt.ylim(ylim)
	#plt.grid()
	plt.title("Trajectory in the atmosphere of "+planets[planet]['name'])
	
	# draw the planet
	plt.gca().add_patch(Circle((0, 0), R, color=planets[planet]['color']))
	
	plt.gca().set_aspect('equal')
	
	# draw the curves
	filtre = h>=0
	
	#fig, ax1 = plt.subplots()
	
	#color = 'tab:blue'
	#ax1.set_xlabel('time (s)')
	#ax1.set_ylabel('altitude (km)', color=color)
	#ax1.plot(t[filtre], h[filtre]/1000, color=color)
	#ax1.tick_params(axis='y', labelcolor=color)

	#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	#color = 'tab:red'
	#ax2.set_ylabel('speed (km/s)', color=color)  # we already handled the x-label with ax1
	#ax2.plot(t[filtre], v[filtre]/1000, color=color)
	#ax2.tick_params(axis='y', labelcolor=color)

	#fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()
