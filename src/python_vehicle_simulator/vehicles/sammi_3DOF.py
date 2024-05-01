#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
otter.py: 
    Class for the Maritime Robotics Otter USV, www.maritimerobotics.com. 
    The length of the USV is L = 2.0 m. The constructors are:

    otter()                                          
        Step inputs for propeller revolutions n1 and n2
        
    otter('headingAutopilot',psi_d,V_current,beta_current,tau_X)  
       Heading autopilot with options:
          psi_d: desired yaw angle (deg)
          V_current: current speed (m/s)
          beta_c: current direction (deg)
          tau_X: surge force, pilot input (N)
        
Methods:
    
[nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) returns 
    nu[k+1] and u_actual[k+1] using Euler's method. The control inputs are:

    u_control = [ n1 n2 ]' where 
        n1: propeller shaft speed, left (rad/s)
        n2: propeller shaft speed, right (rad/s)

u = headingAutopilot(eta,nu,sampleTime) 
    PID controller for automatic heading control based on pole placement.

u = stepInput(t) generates propeller step inputs.

[n1, n2] = controlAllocation(tau_X, tau_N)     
    Control allocation algorithm.
    
References: 
  T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion 
     Control. 2nd. Edition, Wiley. 
     URL: www.fossen.biz/wiley            

Author:     Thor I. Fossen
"""
import numpy as np
import math
try:
    from . import utils, pid
except ImportError:
    import utils, pid
import timeit

# Class Vehicle
class sammi:
    """
    otter()                                           Propeller step inputs
    otter('headingAutopilot',psi_d,V_c,beta_c,tau_X)  Heading autopilot
    
    Inputs:
        psi_d: desired heading angle (deg)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
        tau_X: surge force, pilot input (N)        
    """

    def __init__(
        self, 
        controlSystem="stepInput", 
        r = 0, 
        V_current = 0, 
        beta_current = 0,
        tau_X = 120,
        des_vel=2,
        sample_time=0.01,
        nu=np.array([0, 0, 0, 0, 0, 0], float),
    ):
        
        # Constants
        D2R = math.pi / 180     # deg2rad
        self.g = 9.81           # acceleration of gravity (m/s^2)
        rho = 1025              # density of water (kg/m^3)

        if controlSystem == "headingAutopilot":
            self.controlDescription = (
                "Heading autopilot, psi_d = "
                + str(r)
                + " deg"
                )
        else:
            self.controlDescription = "Step inputs for n1 and n2"
            controlSystem = "stepInput"

        self.ref = r  # Desired heading angle (deg) / psi_d
        self.V_c = V_current  # (ocean) current speed (m/s)
        self.beta_c = beta_current * D2R  # current direction (rad)
        self.controlMode = controlSystem

        # Initialize the Otter USV model
        self.T_n = sample_time  # propeller time constants (s)
        self.L = 2.0    # length (m)
        self.B = 1.7   # beam (m)
        self.nu = nu  # velocity vector p. 22 nu = [u, v, w, p, q, r]
        self.u_actual = np.array([0, 0], float)  # propeller revolution states
        self.name = "SAMMI USV"

        self.controls = [
            "Left propeller shaft speed (rad/s)",
            "Right propeller shaft speed (rad/s)"
        ]
        self.dimU = len(self.controls)

        # Vehicle parameters
        m = 120.0                                 # mass (kg)
        self.mp = 0.                           # Payload (kg)
        self.m_total = m + self.mp
        # self.rp = np.array([0.05, 0, -0.35], float) # location of payload (m) Recomment if using payload
        self.rg = np.array([0.118, 0, -0.024], float)     # CG for hull only (m) (pg. 20) TODO: Should these be reduced to 2D? eg. remove the z axis?
        # rg = (m * rg + self.mp * self.rp) / (m + self.mp)  # CG corrected for payload (recomment if using payload)
        self.S_rg = Smtrx(self.rg)  # p. 24 TODO 3DOF version?
        # self.S_rp = Smtrx(self.rp)  # Recomment if using payload

        R66 = 0.25 * self.L
        T_sway = 1.0        # time constant in sway (s) p. 124 (2011)
        T_yaw = 1.0         # time constant in yaw (s)
        Umax = 1.5   # max forward speed (m/s)

        # Data for one pontoon
        self.B_pont = 0.076  # beam of one pontoon (m)
        y_pont = 0.348      # distance from centerline to waterline centroid (m) (distance between centre of hulls / 2)
        Cw_pont = 0.98      # waterline area coefficient (-) (see notebook)
        Cb_pont = 0.785       # block coefficient, computed from m = 55 kg (see notebook)

        # Inertia dyadic, volume displacement and draft
        nabla = (m + self.mp) / rho  # volume 
        self.T = nabla / (2 * Cb_pont * self.B_pont * self.L)  # draft
        Iz_CG = R66 ** 2 * m
        Iz = Iz_CG - m * (self.S_rg @ self.S_rg)[2][2] # - self.mp * (self.S_rp @ self.S_rp)[2][2]  # TODO is there a faster way to do this?

        # Experimental propeller data including lever arms
        self.k_pos = 24.12 / (rho * (0.076 ** 4) * abs(40.93) * 40.93)  # Calculated at 12V, 1800 PWM (max 1900) (see notebook)
        self.k_neg = 19.12 / (rho * (0.076 ** 4) * abs(40.95) * 40.95)  # Calculated at 12V, 1200 PWM (min 1100) (see notebook)
        self.n_max = 313.635667  # max. prop. rev. (rad/s)
        self.n_min = -311.645991  # min. prop. rev. (rad/s)

        # MRB_CG = [ (m+mp) * I3  O3      (Fossen 2021, Chapter 3) . 64
        #               O3       Ig ]
        MRB = np.zeros((3, 3))
        MRB[0:2, 0:2] = (m + self.mp) * np.identity(2)
        MRB[2][2] = Iz
        MRB[1][2] = m * self.rg[0]
        MRB[2][1] = m * self.rg[0]

        # Hydrodynamic added mass (best practice) p. 116/147 TODO Find where these come from
        # Discussion on symettry and 3DOF on p. 147!
        Xudot = -0.1 * m
        Yvdot = -1.5 * m
        Nrdot = -1.7 * Iz

        # self.MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])
        self.MA = -np.diag([Xudot, Yvdot, Nrdot])

        # System mass matrix
        self.M = MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Hydrostatic quantities (Fossen 2021, Chapter 4)
        Aw_pont = Cw_pont * self.L * self.B_pont  # waterline area, one pontoon p. 80
        I_T = (
            2
            * (1 / 12)
            * self.L
            * self.B_pont ** 3
            * (6 * Cw_pont ** 3 / ((1 + Cw_pont) * (1 + 2 * Cw_pont)))
            + 2 * Aw_pont * y_pont ** 2  # Where does this addition come from??
        )  # Second moment of area transverse p. 80 (or area moment of inertia)
        I_L = 0.8 * 2 * (1 / 12) * self.B_pont * self.L ** 3  # Second moment of area longitudinal p. 81 (or area moment of inertia)
        KB = (1 / 3) * (5 * self.T / 2 - 0.5 * nabla / (self.L * self.B_pont))  # The distance between the keel line and the centre of buoyancy p. 80
        BM_T = I_T / nabla  # BM values The transverse distance between the centre of buoyancy and the metacentre p. 80
        BM_L = I_L / nabla  # The longitudinal distance between the centre of buoyancy and the metacentre p. 80
        KM_T = KB + BM_T    # KM values The transverse distance between the keel and the metacentre p. 81
        KM_L = KB + BM_L    # The longitudinal distance between the keel and the metacentre p. 81
        KG = self.T - self.rg[2] # The distance between the keel and the centre of gravity p. 81

        # Linear damping terms (hydrodynamic derivatives) p. 150/119
        Xu = -24.4 * self.g / Umax   # specified using the maximum speed
        Yv = -self.M[1, 1]  / T_sway # specified using the time constant in sway
        Nr = -self.M[2, 2] / T_yaw  # specified by the time constant T_yaw

        # self.D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr])  # Linear damping for suface vessels p. 150
        self.D = -np.diag([Xu, Yv, Nr])  # Linear damping for suface vessels p. 150

        # Heading autopilot
        self.e_int = 0  # integral state
        self.wn = 2.5   # PID pole placement
        self.zeta = 1

        # SAMMI Control
        speed_pid = [200, 200, 0]
        ang_pid = [150, 0.01, 120]
        self.vel_pid = pid.PID(speed_pid[0], speed_pid[1], speed_pid[2], clamp=313)
        self.ang_pid = pid.PID(ang_pid[0], ang_pid[1], ang_pid[2], clamp=313)
        self.setpoints = [des_vel, self.ref]


    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the Otter USV equations of motion using Euler's method.
        """

        # Input vector
        n = np.array([u_actual[0], u_actual[1]])

        # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[2])  # current surge vel. (beta_c = current direction and eta[5] = yaw angle)
        v_c = self.V_c * math.sin(self.beta_c - eta[2])  # current sway vel.

        nu_c = np.array([u_c, v_c, 0], float)  # current velocity vector 
        Dnu_c = np.array([0, 0, 0], float)  # derivative 
        nu_r = nu - nu_c  # relative velocity vector

        # Rigid body and added mass Coriolis and centripetal matrices
        CRB = np.zeros((3, 3))
        CRB = np.array([[0, -self.m_total * nu[2], -self.m_total * self.rg[2] * nu[2]],
                        [self.m_total * nu[2], 0, 0],
                        [self.m_total * self.rg[2] * nu[2], 0, 0]], float)

        CA = m2c(self.MA, nu_r)  # p. 156
        # Uncomment to cancel the Munk moment in yaw, if stability problems
        # CA[5, 0] = 0  
        # CA[5, 1] = 0 
        # CA[0, 5] = 0
        # CA[1, 5] = 0
        C = CRB + CA

        # Control forces and moments - with propeller revolution saturation
        thrust = np.zeros(2)
        for i in range(0, 2):
            n[i] = sat(n[i], self.n_min, self.n_max)  # saturation, physical limits
            if n[i] > 0:  # positive thrust (Why are there different K_ts for forward/reverse)
                thrust[i] = 997 * np.power(0.076, 4) * self.k_pos * abs(n[i]) * n[i] * 2
            else:  # negative thrust
                thrust[i] = 997 * np.power(0.076, 4) * self.k_neg * abs(n[i]) * n[i] * 2
        # Control forces and moments
        # Mousazadeh et al. thrust (Eq. 5)
        thrust_distance = 1.4  # distance between the two thrusters (m)
        tau = np.array(
            [
                thrust[0] + thrust[1],
                0,
                (thrust[0] - thrust[1]) * thrust_distance / 2
            ]
        )

        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = -np.matmul(self.D, nu_r)
        tau_damp[2] = tau_damp[2] - 10 * self.D[2, 2] * abs(nu_r[2]) * nu_r[2]

        # State derivatives (with dimension)
        # tau_crossflow = crossFlowDrag(self.L, self.B_pont, self.T, nu_r)  # Recomment and solve this if using current velocity
        sum_tau = (
            tau
            + tau_damp
            - np.matmul(C, nu_r)
        )

        nu_dot = Dnu_c + np.matmul(self.Minv, sum_tau)  # USV dynamics
        n_dot = (u_control - n) / self.T_n  # propeller dynamics

        # Forward Euler integration [k+1]
        nu = nu + sampleTime * nu_dot
        self.nu = nu
        n = n + sampleTime * n_dot

        u_actual = np.array(n, float)
        return nu, u_actual

    def set_setpoint(self, vel, ang):
        self.setpoints = [vel, ang]

    def update(self, eta, dt):
        velocity = math.sqrt(self.nu[0] ** 2 + self.nu[1] ** 2) 
        bearing = eta[2]

        vel_val = self.vel_pid.update(
            self.setpoints[0] - velocity, dt)

        ang_dif = utils.heading_error(bearing,
                                      np.deg2rad(self.setpoints[1]))
        ang_val = self.ang_pid.update(ang_dif, dt)

        if np.rad2deg(bearing) > 90:
            pass

        return self.thruster_control(vel_val, ang_val)
        # return np.array([313, 313], float)

    def thruster_control(self, vel_val, ang_val):
        vel_r = (vel_val - ang_val) / 2
        vel_l = (vel_val + ang_val) / 2
        return np.array([vel_l, vel_r], float)

def simulate(initial_state, initial_velocities, des_ang, des_vel, time, sample_time=0.01, tracking=["final"]):
    sample_count = int(time/sample_time)
    vehicle = sammi(r=des_ang, des_vel=des_vel, sample_time=sample_time, nu=initial_velocities)
    eta = initial_state
    nu = vehicle.nu
    u_actual = vehicle.u_actual
    if "eta" in tracking:
        eta_hist = np.zeros((sample_count, 6))
    if "nu" in tracking:
        nu_hist = np.zeros((sample_count, 6))
    if "u_actual" in tracking:
        u_actual_hist = np.zeros((sample_count, 2))

    for i in range(0, sample_count):
        t = i * sample_time
        u_control = vehicle.update(eta, sample_time)
        [nu, u_actual] = vehicle.dynamics(eta, nu, u_actual, u_control, sample_time)
        eta = attitudeEuler(eta, nu, sample_time)

        if "eta" in tracking:
            eta_hist[i] = eta
        if "nu" in tracking:
            nu_hist[i] = nu
        if "u_actual" in tracking:
            u_actual_hist[i] = u_actual
    
    return_data = []
    if "eta" in tracking:
        return_data.append(eta_hist)
    if "nu" in tracking:
        return_data.append(nu_hist)
    if "u_actual" in tracking:
        return_data.append(u_actual_hist)
    if "final" in tracking:
        return_data.append([eta, nu, u_actual])
    return return_data

def Rzyx(psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """

    cpsi = math.cos(psi)
    spsi = math.sin(psi)
    
    R = np.array([
        [ cpsi, -spsi],
        [ spsi, cpsi ]
    ])

    return R

def attitudeEuler(eta,nu,sampleTime):
    """
    eta = attitudeEuler(eta,nu,sampleTime) computes the generalized 
    position/Euler angles eta[k+1]
    """
   
    # p_dot   = np.matmul( Rzyx(eta[3], eta[4], eta[5]), nu[0:3] )
    p_dot = np.matmul(Rzyx(eta[2]), nu[0:2])
    # v_dot   = np.matmul( Tzyx(eta[3], eta[4]), nu[3:6] )  # Not needed for 3DOF

    # Forward Euler integration
    eta[0:2] = eta[0:2] + sampleTime * p_dot
    # eta[3:6] = eta[3:6] + sampleTime * v_dot

    return eta

def Smtrx(a):
    """
    S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
    The cross product satisfies: a x b = S(a)b. 
    """
 
    S = np.array([ 
        [ 0, -a[2], a[1] ],
        [ a[2],   0,     -a[0] ],
        [-a[1],   a[0],   0 ]  ])

    return S

def m2c(M, nu):
    """
    C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)
    """

    M = 0.5 * (M + M.T)     # systematization of the inertia matrix

    #C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
    #      0             0             M(1,1)*nu(1)
    #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]    
    C = np.zeros( (3,3) ) 
    C[0,2] = -M[1,1] * nu[1] - M[1,2] * nu[2]
    C[1,2] =  M[0,0] * nu[0] 
    C[2,0] = -C[0,2]       
    C[2,1] = -C[1,2]
        
    return C

def sat(x, x_min, x_max):
    """
    x = sat(x,x_min,x_max) saturates a signal x such that x_min <= x <= x_max
    """
    if x > x_max:
        x = x_max 
    elif x < x_min:
        x = x_min
        
    return x    

if __name__ == "__main__":
    try:
        timer = input("Do you want to time the simulation? (y/n): ")
        if timer == "n":
            des_vel = float(input("Enter a desired velocity: "))
            des_ang = float(input("Enter a desired angle: "))
            time = float(input("Enter a time for the simulation: "))
        elif timer == "y":
            des_vel = 0.5
            des_ang = 0
            time = 5
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            exit()
    except ValueError:
        print("Invalid input. Please enter valid numeric values for desired velocity and angle.")
        exit()

    # initial_state = np.array([0, 0, 0, 0, 0, np.deg2rad(90)], float)
    initial_state = np.array([0, 0, np.deg2rad(90)], float)
    # initial_velocities = np.array([0, 0, 0, 0, 0, 0], float)
    initial_velocities = np.array([0, 0, 0], float)
    sample_time = 0.01
    tracking = ["eta"]
    if timer == "y":
        iterations = 1000
        total_time = timeit.timeit("simulate(initial_state, initial_velocities, des_ang, des_vel, time, sample_time, tracking)", setup="from __main__ import simulate, initial_state, initial_velocities, des_ang, des_vel, time, sample_time, tracking", number=iterations)
        print("Average time for {} iterations: ".format(iterations), total_time/iterations)
        print("Average time per simulation second: ", (total_time/iterations)/time)
    else:
        data = simulate(initial_state, initial_velocities, des_ang, des_vel, time, sample_time, tracking)
        [eta, nu, u_actual] = data[0]
        print("Simulation complete", eta, nu, u_actual)


