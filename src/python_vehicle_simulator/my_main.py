#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from vehicles import sammi_6DOF, sammi_3DOF
# from python_vehicle_simulator.lib import *
from lib.plotTimeSeries import *

# Simulation parameters: 
sampleTime = 0.01                  # sample time [seconds]
time = 10                           # simulation time [seconds]
N = int(time / sampleTime)         # number of samples

# 3D plot and animation parameters where browser = {firefox,chrome,safari,etc.}
numDataPoints = 50                  # number of 3D data points
FPS = 10                            # frames per second (animated GIF)
filename = '3D_animation.gif'       # data file for animated GIF
browser = 'safari'                  # browser for visualization of animated GIF

def main():    
    DOF = 3                     # degrees of freedom
    t = 0                       # initial simulation time

    # Initial state vectors
    if DOF == 6:
        vehicle = sammi_6DOF.sammi(r=0, des_vel=0.5, sample_time=sampleTime, nu=np.array([0, 0, 0, 0, 0, 0], float)) 
        eta = np.array([0, 0, 0, 0, 0, 0], float)    # position/attitude, user editable
    if DOF == 3:
        vehicle = sammi_3DOF.sammi(r=0, des_vel=0.5, sample_time=sampleTime, nu=np.array([0, 0, 0], float))
        eta = np.array([0, 0, 0], float)
    nu = vehicle.nu                              # velocity, defined by vehicle class
    u_actual = vehicle.u_actual                  # actual inputs, defined by vehicle class
    
    # Initialization of table used to store the simulation data
    simData = np.empty( [0, 2*DOF + 2 * vehicle.dimU], float)

    # Simulator for-loop
    for i in range(0,N+1):
        
        t = i * sampleTime      # simulation time
        
        # u_control = vehicle.headingAutopilot(eta,nu,sampleTime)     
        u_control = vehicle.update(eta, sampleTime)
        # print(u_control)
        # Store simulation data in simData
        signals = np.append( np.append( np.append(eta,nu),u_control), u_actual )
        simData = np.vstack( [simData, signals] ) 

        # Propagate vehicle and attitude dynamics
        [nu, u_actual]  = vehicle.dynamics(eta,nu,u_actual,u_control,sampleTime)
        eta = sammi_3DOF.attitudeEuler(eta,nu,sampleTime)

    # Store simulation time vector
    simTime = np.arange(start=0, stop=t+sampleTime, step=sampleTime)[:, None]
    
    plotVehicleStates(simTime, simData, 1, DOF=DOF)                    
    plotControls(simTime, simData, vehicle, 2, DOF=DOF)
    # plot3D(simData, numDataPoints, FPS, filename, 3)
    
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()