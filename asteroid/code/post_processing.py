# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:22:12 2020

@author: Piggy
"""

""" Import packages """

import os #path handling
import numpy as np #import numpy drives sklearn to use numpy arrays instead of python lists
import pandas as pd #CSV and dataframe handling

from pykep import AU, DEG2RAD, MU_SUN, MU_EARTH, epoch, EARTH_RADIUS
import pykep as pk
from pykep.planet import jpl_lp, keplerian

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import spiceypy

import NEA_database as db

MU_MOON = 4902.7779 
R_MOON = 1737.1e3
g0 = 9.80665

""" Functions """

def mass_tsiol(dv, isp, m_init):
    return m_init*np.exp(-dv/isp/g0)

def get_totalDV(z, isp):
    return isp*g0*np.log(z['m'][0]/z['m'][len(z)-1])

def plot_traj(z):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(z['x'], z['y'], z['z'], color='blue')
#    plt.xlabel('X')
#    plt.ylabel('Y')
#    plt.zlabel('Z')
#    plt.show()
    return

def change_geocentric_position(time, pos_helio):
    earth = pk.planet.jpl_lp('earth')
    pos_earth2sun, vel_earth2sun = earth.eph(time)
    pos_earth2sun = np.array(pos_earth2sun)
    vel_earth2sun = np.array(vel_earth2sun)
    
    pos_geo = pos_helio - pos_earth2sun
#    print(np.linalg.norm(pos_geo))
    return pos_geo

def change_geocentric_velocity(time, vel_helio):
    earth = pk.planet.jpl_lp('earth')
    pos_earth2sun, vel_earth2sun = earth.eph(time)
    
    vel_geo = vel_helio - vel_earth2sun
    return vel_geo

def flyby_rel_speed(planet, kin):
    meeting_time = kin['time'][len(kin)-1]
    pos_pl, vel_pl = planet.eph(pk.epoch(meeting_time, 'mjd2000'))   #position, velocity
    vel_pl = np.array(vel_pl)
    print(vel_pl)
    rel_vel = np.empty([1,3])
    rel_vel[0,0] = kin['vx'][len(kin)-1]-vel_pl[0]
    rel_vel[0,1] = kin['vy'][len(kin)-1]-vel_pl[1]
    rel_vel[0,2] = kin['vz'][len(kin)-1]-vel_pl[2]
    return np.linalg.norm(rel_vel,axis=1)

def rel_speed(planet, kin, iteration):
    meeting_time = kin['time'][iteration]
    print(meeting_time)
    pos_pl, vel_pl = planet.eph(pk.epoch(meeting_time, 'mjd2000'))   #position, velocity
    vel_pl = np.array(vel_pl)
    print(vel_pl)
    rel_vel = np.empty([1,3])
    rel_vel[0,0] = kin['vx'][iteration]-vel_pl[0]
    rel_vel[0,1] = kin['vy'][iteration]-vel_pl[1]
    rel_vel[0,2] = kin['vz'][iteration]-vel_pl[2]
    return np.linalg.norm(rel_vel,axis=1)



""" Data reading """

filename = '150kg_safran_traj.csv'
path = 'D:/Utilisateurs/GitHub/piggy_code/asteroid/results/target_2011BT59/fromMoon'
path = os.path.join(path, filename)

#t, x, y, z, vx, vy, vz, m, u, ux, uy, uz
#time in days
#in meter
kin = pd.read_csv(path,header=None,sep='\s+',names=['time','x','y','z','vx','vy','vz','m','T','Tx','Ty','Tz']) 
kin['v'] = [np.sqrt(kin['vx'][i]**2+kin['vy'][i]**2+kin['vz'][i]**2) for i in range(len(kin['vx']))]

#earth = pk.planet.jpl_lp('earth')
pk.util.load_spice_kernel('de430.bsp')
earth = pk.planet.spice('EARTH', 'SUN', 'ECLIPJ2000', 'NONE', MU_SUN, MU_EARTH, EARTH_RADIUS, EARTH_RADIUS * 1.05)

print('total DV =', get_totalDV(kin, 3000), 'm/s')

rel_earth = []
for i in range(len(kin)):
    time = kin['time'][i]
    r,v = earth.eph(epoch(time,'mjd2000'))
    rel_earth.append(np.abs(kin))

#pos_helio = np.array([kin['x'][0], kin['y'][0], kin['z'][0]])
#vel_helio = np.array([kin['vx'][0], kin['vy'][0], kin['vz'][0]])
##pos_geo = change_geocentric_position(kin['time'][0], pos_helio)
#
#num = 80
#traj_to_inf = np.empty([num,3])
#vel_to_inf = np.empty([num,3])
#for i in range(num):
#    pos_helio = np.array([kin['x'][i], kin['y'][i], kin['z'][i]])
#    vel_helio = np.array([kin['vx'][i], kin['vy'][i], kin['vz'][i]])
#    time = kin['time'][i]
#    traj_to_inf[i] = change_geocentric_position(time, pos_helio)
#    vel_to_inf[i] = change_geocentric_velocity(time, vel_helio)
#
#index=1
#
#instant_to_inf = kin['time'][index]
#instant_of_depart = instant_to_inf-30 #catch the liberation point with 30 days in advance from gto
#
#earth = jpl_lp('earth')
##gto_object = keplerian(pk.epoch_from_iso_string('20200520T235954'), #MJD = JD - 2400000.5
#gto_object = keplerian(pk.epoch(instant_of_depart, 'mjd2000'),  
#                                 [24630e3,           #a
#                                  0.716,                 #e
#                                  1e-6 * DEG2RAD,       #i
#                                  170 * DEG2RAD,     #W
#                                  0. * DEG2RAD,     #w
#                                  200 * DEG2RAD],    #M anomalie 
#                                 MU_EARTH,    #mu_central_body
#                                 0.,        #mu_self
#                                 0., 0., #radius, safe_radius : safe_radius must be greater than radius km
#                                 "GTO")  #body name
#
##state at departure pt_from_gto, mass = masse initiale, 
#pt_from_gto = gto_object.eph(pk.epoch(instant_of_depart, 'mjd2000')) #at instant instant_of_depart
##traj_to_inf, vel_to_inf at instant instant_to_inf, mass = kin['mass'][index]
#
#fig = plt.figure()
#ax = fig.gca(projection = '3d')
#ax = pk.orbit_plots.plot_planet(gto_object, axes = ax, color='b', N=100)
#ax.scatter(0,0,0, color='r')
##ax.plot(traj_to_inf[:,0], traj_to_inf[:,1], traj_to_inf[:,2])
#ax.scatter(traj_to_inf[index,0], traj_to_inf[index,1], traj_to_inf[index,2], color='black')
##index=2
##ax.scatter(traj_to_inf[index,0], traj_to_inf[index,1], traj_to_inf[index,2], color='black')
##    ax.quiver(traj_to_inf[i,0], traj_to_inf[i,1], traj_to_inf[i,2], 
##              vel_to_inf[i,0], vel_to_inf[i,1], vel_to_inf[i,2]/100,
##              length=1.0e2,
##              normalize=True,
##              color='black')
#ax.set_xlabel('x')
#ax.set_ylabel('y')






