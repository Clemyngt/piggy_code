#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.patches import Circle
from matplotlib import ticker, cm


G = 6.67e-11
B = 1

planet = "earth"

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
		'R': 6371e3,
		'M': 5.97e24,
		'rho0': 1.39,
		'hs': 7.9e3,
		'color':'#e5a77f'
	},
	'venus':{
		'name': 'Venus',
		'R': 6371e3,
		'M': 5.97e24,
		'rho0': 1.39,
		'hs': 7.9e3,
		'color':'#fff000'
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
		return [0, 0, 0, 0]
		
	dydt = [ar, vr, atheta, vtheta]
	return dydt

if __name__ == "__main__":
	from scipy.integrate import odeint
	
	x0 = R+350e3
	y0 = 0
	vx0 = 0
	vy0 = -7.645e3
	
	r0 = np.hypot(x0, y0)
	theta0 = np.arctan2(y0, x0)
	vr0 = (x0*vx0+y0*vy0)/r0
	vtheta0 = (x0*vy0 - y0*vx0)/(r0*r0)
	
	
	y0 = [vr0, r0, vtheta0, theta0]
	t = np.linspace(0, 20000, 10000)
	
	sol = odeint(equations, y0, t)
	
	vr, r, vtheta, theta = sol.T
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	
	h = r-R;
	v = np.hypot(vr, vtheta*r)
	
	# plot limits
	border = 400e3
	xlim = [np.min(x)-border, np.max(x)+border]
	ylim = [np.min(y)-border, np.max(y)+border]
	
	# the atmosphere
	xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 1000), np.linspace(ylim[0], ylim[1], 1000))
	rr = np.hypot(xx, yy)
	zz = np.where(rr < R+1e3, 0, rho(rr))
	
	#locator.nonsingular(1e-10, 1)
	plt.contourf(xx, yy, zz, locator=ticker.LogLocator(numticks=200), cmap='gray', vmin=1e-50)
	cbar = plt.colorbar()
	cbar.set_label('atmospheric density ($kg.m^{-3}$)', rotation=270)
	
	# draw the planet
	plt.gca().add_patch(Circle((0, 0), R, color=planets[planet]['color']))
	
	# draw the trajectory
	plt.scatter(x, y, c=t, label='trajectory', cmap='copper', s=0.5)
	#plt.plot(t, r, 'g', label='r(t)')
	plt.legend()
	plt.xlim(xlim)
	plt.ylim(ylim)
	#plt.grid()
	plt.title("Trajectory in the atmosphere of "+planets[planet]['name'])
	
	plt.gca().set_aspect('equal')
	
	# draw the curves
	filtre = h>=0
	
	fig, ax1 = plt.subplots()
	
	color = 'tab:blue'
	ax1.set_xlabel('time (s)')
	ax1.set_ylabel('altitude (km)', color=color)
	ax1.plot(t[filtre], h[filtre]/1000, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:red'
	ax2.set_ylabel('speed (km/s)', color=color)  # we already handled the x-label with ax1
	ax2.plot(t[filtre], v[filtre]/1000, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()
