#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
otter.py: 
    Class for the Maritime Robotics Otter USV, www.maritimerobotics.com. 
    The constructors are:
        
    sammi('headingAutopilot',psi_d,V_current,beta_current,tau_X)  
       Heading autopilot with options:
          psi_d: desired yaw angle (deg)
          V_current: current speed (m/s)
          beta_c: current direction (deg)
          tau_X: surge force, pilot input (N)
          des_vel: desired velocity (m/s)
          sample_time: sample time (s)
          nu: initial velocity vector [u v w p q r] (m/s, rad/s)
        
Methods:
    
    [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) returns 
        nu[k+1] and u_actual[k+1] using Euler's method. The control inputs are:

    u_control = [ n1 n2 ]' where 
        n1: propeller shaft speed, left (rad/s)
        n2: propeller shaft speed, right (rad/s)

    u = update(eta, dt)
        PID controller for velocity and angle control.
    
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
import matplotlib.pyplot as plt
import copy

# Class Vehicle
class sammi:
    """
    sammi('',psi_d,V_c,beta_c,tau_X,des_vel,sample_time,nu)  Heading autopilot
    
    Inputs:
        psi_d: desired heading angle (deg)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
        tau_X: surge force, pilot input (N)   
        des_vel: desired velocity (m/s)
        sample_time: sample time (s)
        nu: initial velocity vector [u v w p q r] (m/s, rad/s)     
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
        nu=np.array([0, 0, 0], float),
        test_integrals=False
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

        if nu[0] != 0 and not test_integrals:
            integral_start, control_start = self.find_starting_values(nu[0])
        else:
            integral_start = 0
            control_start = [0, 0]
        # Initialize the Otter USV model
        self.T_n = sample_time  # propeller time constants (s)
        self.L = 2.0    # length (m)
        self.B = 1.7   # beam (m)
        self.nu = nu  # velocity vector p. 22 nu = [u, v, r]
        self.u_actual = np.array([control_start[0], control_start[1]], float)  # propeller revolution states
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
        self.B_pont = 0.3  # beam of one pontoon (m)
        y_pont = 0.7      # distance from centerline to waterline centroid (m) (distance between centre of hulls / 2)
        Cw_pont = 0.98      # waterline area coefficient (-) (see notebook)
        Cb_pont = 0.785       # block coefficient, computed from m = 55 kg (see notebook)

        # Inertia dyadic, volume displacement and draft
        nabla = (m + self.mp) / rho  # volume 
        self.T = nabla / (2 * Cb_pont * self.B_pont * self.L)  # draft
        Iz_CG = R66 ** 2 * m
        Iz = Iz_CG - m * (self.S_rg @ self.S_rg)[2][2] # - self.mp * (self.S_rp @ self.S_rp)[2][2]  # TODO is there a faster way to do this?

        # Experimental propeller data including lever arms
        self.prop_diameter = 0.076
        self.k_pos = 24.12 / (rho * (self.prop_diameter ** 4) * abs(40.93) * 40.93)  # Calculated at 12V, 1800 PWM (max 1900) (see notebook)
        self.k_neg = 19.12 / (rho * (self.prop_diameter ** 4) * abs(40.95) * 40.95)  # Calculated at 12V, 1200 PWM (min 1100) (see notebook)
        # self.n_max = 313.635667  # max. prop. rev. (rad/s)
        self.n_max = 49.9 # rps
        # self.n_min = -311.645991  # min. prop. rev. (rad/s)
        self.n_min = -49.6 # rps

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

        self.vel_pid = pid.PID(speed_pid[0], speed_pid[1], speed_pid[2], clamp=313, starting_integral=integral_start)
        self.ang_pid = pid.PID(ang_pid[0], ang_pid[1], ang_pid[2], clamp=313)
        self.setpoints = [des_vel, self.ref]
        self.test_integrals = test_integrals

    def find_starting_values(self, cur_vel):
        vel_dict = {0.0: [0.0, 0.0], 0.01: [0.05263609990543663, 0.0], 0.02: [0.07444255184325736, 0.0], 0.03: [0.09117325373080157, 0.0], 0.04: [0.10527781677842944, 0.0], 0.05: [0.11770417945040745, 0.0], 0.06: [0.12893846891229171, 0.0], 0.07: [0.1392694640156822, 0.0], 0.08: [0.14888531972125582, 0.0], 0.09: [0.15791672882230903, 0.0], 0.1: [0.166458847923271, 0.0], 0.11: [0.17458351256713045, 0.0], 0.12: [0.1823465318225966, 0.0], 0.13: [0.1897922877112785, 0.0], 0.14: [0.19695676500524642, 0.0], 0.15: [0.20386962031600964, 0.0], 0.16: [0.21055563847622755, 0.0], 0.17: [0.21703578437756618, 0.0], 0.18: [0.22332797967693163, 0.0], 0.19: [0.22944768752971123, 0.0], 0.2: [0.23540836034188092, 0.0], 0.21: [0.24122178783217352, 0.0], 0.22: [0.2468983712603821, 0.0], 0.23: [0.2524473421074084, 0.0], 0.24: [0.25787693837012465, 0.0], 0.25: [0.2631945480992999, 0.0], 0.26: [0.26840682732604426, 0.0], 0.27: [0.2735197977501178, 0.0], 0.28: [0.2785389282798605, 0.0], 0.29: [0.28346920357124655, 0.0], 0.3: [0.2883151820132563, 0.0], 0.31: [0.29308104508028915, 0.0], 0.32: [0.2977706395724224, 0.0], 0.33: [0.30238751395747576, 0.0], 0.34: [0.3069349497913094, 0.0], 0.35000000000000003: [0.31141598900731315, 0.0], 0.36: [0.31583345772006394, 0.0], 0.37: [0.32018998707239127, 0.0], 0.38: [0.32448803156268224, 0.0], 0.39: [0.32872988521494223, 0.0], 0.4: [0.3329176958940953, 0.0], 0.41000000000000003: [0.33705347802008584, 0.0], 0.42: [0.3411391238944005, 0.0], 0.43: [0.34517641381971803, 0.0], 0.44: [0.3491670251662416, 0.0], 0.45: [0.3531125405156888, 0.0], 0.46: [0.3570144549950946, 0.0], 0.47000000000000003: [0.3608741828968161, 0.0], 0.48: [0.36469306366785303, 0.0], 0.49: [0.3684723673404073, 0.0], 0.5: [0.3722132994661043, 0.0], 0.51: [0.3759170056082101, 0.0], 0.52: [0.379584575439312, 0.0], 0.53: [0.38321704648598576, 0.0], 0.54: [0.38681540755694005, 0.0], 0.55: [0.3903806018867246, 0.0], 0.56: [0.3939135300233299, 0.0], 0.5700000000000001: [0.39741505248473025, 0.0], 0.58: [0.40088599220656296, 0.0], 0.59: [0.40432713680068577, 0.0], 0.6: [0.4077392406421545, 0.0], 0.61: [0.4111230268003013, 0.0], 0.62: [0.41447918882790974, 0.0], 0.63: [0.4178083924210274, 0.0], 0.64: [0.4211112769606684, 0.0], 0.65: [0.4243884569465179, 0.0], 0.66: [0.4276405233317452, 0.0], 0.67: [0.4308680447671329, 0.0], 0.68: [0.43407156876194464, 0.0], 0.6900000000000001: [0.43725162276824125, 0.0], 0.7000000000000001: [0.44040871519472236, 0.0], 0.71: [0.4435433363556178, 0.0], 0.72: [0.44665595935964086, 0.0], 0.73: [0.4497470409435636, 0.0], 0.74: [0.4528170222545789, 0.0], 0.75: [0.45586632958524076, 0.0], 0.76: [0.4588953750644462, 0.0], 0.77: [0.4619045573076358, 0.0], 0.78: [0.46489426202911405, 0.0], 0.79: [0.46786486261914983, 0.0], 0.8: [0.4708167206883089, 0.0], 0.81: [0.473750186581256, 0.0], 0.8200000000000001: [0.47666559986210544, 0.0], 0.8300000000000001: [0.4795632897732203, 0.0], 0.84: [0.48244357566925034, 0.0], 0.85: [0.4853067674281574, 0.0], 0.86: [0.4881531658418309, 0.0], 0.87: [0.4909830630059126, 0.0], 0.88: [0.49379674458883177, 0.0], 0.89: [0.49893056564181754, 0.0]}
        control_dict = {0.0: [-2.71925883, -2.71925883], 0.01: [5.26364061, 5.26364061], 0.02: [7.44425889, 7.44425889], 0.03: [9.11732594, 9.11732594], 0.04: [10.52778181, 10.52778181], 0.05: [11.77041799, 11.77041799], 0.06: [12.89384691, 12.89384691], 0.07: [13.92694641, 13.92694641], 0.08: [14.88853198, 14.88853198], 0.09: [15.79167289, 15.79167289], 0.1: [16.64588479, 16.64588479], 0.11: [17.45835126, 17.45835126], 0.12: [18.23465318, 18.23465318], 0.13: [18.97922877, 18.97922877], 0.14: [19.6956765, 19.6956765], 0.15: [20.38696203, 20.38696203], 0.16: [21.05556385, 21.05556385], 0.17: [21.70357844, 21.70357844], 0.18: [22.33279797, 22.33279797], 0.19: [22.94476875, 22.94476875], 0.2: [23.54083603, 23.54083603], 0.21: [24.12217878, 24.12217878], 0.22: [24.68983713, 24.68983713], 0.23: [25.24473421, 25.24473421], 0.24: [25.78769384, 25.78769384], 0.25: [26.31945481, 26.31945481], 0.26: [26.84068273, 26.84068273], 0.27: [27.35197978, 27.35197978], 0.28: [27.85389283, 27.85389283], 0.29: [28.34692036, 28.34692036], 0.3: [28.8315182, 28.8315182], 0.31: [29.30810451, 29.30810451], 0.32: [29.77706396, 29.77706396], 0.33: [30.2387514, 30.2387514], 0.34: [30.69349498, 30.69349498], 0.35000000000000003: [31.1415989, 31.1415989], 0.36: [31.58334577, 31.58334577], 0.37: [32.01899871, 32.01899871], 0.38: [32.44880316, 32.44880316], 0.39: [32.87298852, 32.87298852], 0.4: [33.29176959, 33.29176959], 0.41000000000000003: [33.7053478, 33.7053478], 0.42: [34.11391239, 34.11391239], 0.43: [34.51764138, 34.51764138], 0.44: [34.91670252, 34.91670252], 0.45: [35.31125405, 35.31125405], 0.46: [35.7014455, 35.7014455], 0.47000000000000003: [36.08741829, 36.08741829], 0.48: [36.46930637, 36.46930637], 0.49: [36.84723673, 36.84723673], 0.5: [37.22132995, 37.22132995], 0.51: [37.59170056, 37.59170056], 0.52: [37.95845754, 37.95845754], 0.53: [38.32170465, 38.32170465], 0.54: [38.68154076, 38.68154076], 0.55: [39.03806019, 39.03806019], 0.56: [39.391353, 39.391353], 0.5700000000000001: [39.74150525, 39.74150525], 0.58: [40.08859922, 40.08859922], 0.59: [40.43271368, 40.43271368], 0.6: [40.77392406, 40.77392406], 0.61: [41.11230268, 41.11230268], 0.62: [41.44791888, 41.44791888], 0.63: [41.78083924, 41.78083924], 0.64: [42.1111277, 42.1111277], 0.65: [42.43884569, 42.43884569], 0.66: [42.76405233, 42.76405233], 0.67: [43.08680448, 43.08680448], 0.68: [43.40715688, 43.40715688], 0.6900000000000001: [43.72516228, 43.72516228], 0.7000000000000001: [44.04087152, 44.04087152], 0.71: [44.35433364, 44.35433364], 0.72: [44.66559594, 44.66559594], 0.73: [44.97470409, 44.97470409], 0.74: [45.28170223, 45.28170223], 0.75: [45.58663296, 45.58663296], 0.76: [45.88953751, 45.88953751], 0.77: [46.19045573, 46.19045573], 0.78: [46.4894262, 46.4894262], 0.79: [46.78648626, 46.78648626], 0.8: [47.08167207, 47.08167207], 0.81: [47.37501866, 47.37501866], 0.8200000000000001: [47.66655999, 47.66655999], 0.8300000000000001: [47.95632898, 47.95632898], 0.84: [48.24435757, 48.24435757], 0.85: [48.53067674, 48.53067674], 0.86: [48.81531658, 48.81531658], 0.87: [49.0983063, 49.0983063], 0.88: [49.37967426, 49.37967426], 0.89: [49.66471063, 49.66471063]}
        try:
            return vel_dict[cur_vel][0], control_dict[cur_vel]
        except KeyError:
            if cur_vel > list(vel_dict.keys())[-1]:
                return vel_dict[list(vel_dict.keys())[-1]][0], control_dict[list(vel_dict.keys())[-1]]
            for key in vel_dict:
                if cur_vel == key:
                    return vel_dict[key][0], control_dict[key]
                elif cur_vel < key:
                    return vel_dict[key][0] + (cur_vel - key) * vel_dict[key-list(vel_dict.keys())[1]][0], control_dict[key] + (cur_vel - key) * control_dict[key-list(control_dict.keys())[1]]

    
    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the SAMMI USV equations of motion using Euler's method.
        """

        # Input vector
        n = np.array([u_actual[0], u_actual[1]])

        # Current velocities (Recomment if current != 0)
        # u_c = self.V_c * math.cos(self.beta_c - eta[2])  # current surge vel. (beta_c = current direction and eta[5] = yaw angle)
        # v_c = self.V_c * math.sin(self.beta_c - eta[2])  # current sway vel.
        # nu_c = np.array([u_c, v_c, 0], float)  # current velocity vector 
        Dnu_c = np.array([0, 0, 0], float)  # derivative 

        nu_r = nu # - nu_c  # relative velocity vector

        # Rigid body and added mass Coriolis and centripetal matrices
        # CRB = np.zeros((3, 3))
        mr = self.m_total * nu[2]
        mxgr = mr * self.rg[2]
        CRB = np.array([[0,    -mr, -mxgr],
                        [mr,   0,   0],
                        [mxgr, 0,   0]], float)

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
                thrust[i] = 1025 * np.power(self.prop_diameter, 4) * self.k_pos * abs(n[i]) * n[i] * 2
            else:  # negative thrust
                thrust[i] = 1125 * np.power(self.prop_diameter, 4) * self.k_neg * abs(n[i]) * n[i] * 2
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
        if self.test_integrals:
            return nu, u_actual, [self.vel_pid.integral, self.ang_pid.integral]
        return nu, u_actual

    def update(self, eta, dt):
        """
        Updates the control of the vehicle based on the current state.

        Args:
            eta (list): The current state of the vehicle, including position and heading.
            dt (float): The time step for the update.

        Returns:
            tuple: The control signals for the thrusters.

        """
        velocity = math.sqrt(self.nu[0] ** 2 + self.nu[1] ** 2) 
        bearing = eta[2]

        vel_val = self.vel_pid.update(
            self.setpoints[0] - velocity, dt)

        ang_dif = utils.heading_error(bearing,
                                      np.deg2rad(self.setpoints[1]))
        ang_val = self.ang_pid.update(ang_dif, dt)

        return self.thruster_control(vel_val, ang_val)

    def thruster_control(self, vel_val, ang_val):
        """
        Controls the thrusters of the vehicle based the output of the PIDs.

        Parameters:
        - vel_val (float): The velocity PID output.
        - ang_val (float): The angle PID output.

        Returns:
        - numpy.ndarray: An array containing the left and right thruster values.

        """
        vel_r = (vel_val - ang_val) / 2
        vel_l = (vel_val + ang_val) / 2
        return np.array([vel_l, vel_r], float)

def simulate(initial_state, initial_velocities, des_ang, des_vel, end_condition, sample_time=0.01, tracking=[], test_integrals=False):
    """
    Simulates the behavior of a vehicle over a given time period.

    Args:
        initial_state (list): The initial state of the vehicle, represented as [x, y, yaw].
        initial_velocities (list): The initial velocities of the vehicle, represented as [u, v, w].
        des_ang (float): The desired angle of the vehicle.
        des_vel (float): The desired velocity of the vehicle.
        end_condition (tuple): The end condition for the simulation. It can be one of the following:
            - ("time", end time): The simulation will end after the specified time (in seconds) has elapsed. (s)
            - ("vel", [desired velocity, threshold]): The simulation will end when the vehicle's velocity magnitude is greater than or equal to the specified value. (m/s)
            - ("ang", [desired angle, threshold]): The simulation will end when the vehicle's yaw angle is within 0.01 degrees of the specified value. (deg)
            - ("vel ang", [velocity, angle, velocity threshold, angle threshold]): The simulation will end when both the velocity magnitude and yaw angle meet the specified conditions. (m/s, deg)
        sample_time (float, optional): The time interval between each sample. Defaults to 0.01.
        tracking (list, optional): The variables to track during the simulation. Can include "eta" (vehicle state),
            "nu" (vehicle velocities), and "u_actual" (actual control inputs). Defaults to an empty list.
        test_integrals (bool, optional): Whether to return the integrals of the PID controllers. Defaults to False.

    Returns:
        list: A list containing the tracked variables over the simulation time. The variables in the list
            are determined by the `tracking` parameter. The last element of the list contains
            the final state of the vehicle, represented as [x, y, yaw], the final velocities of the vehicle,
            represented as [u, v, w], and the final actual control inputs of the vehicle, represented as [n1, n2].
            e.g. tracking = ["eta, "nu", "u_actual"]
                 return = [eta_hist, nu_hist, u_actual_hist, [eta, nu, u_actual]]
    """

    if end_condition[0] == "time":
        sample_count = int(end_condition[1]/sample_time)
    else:
        sample_count = 99999999999

    vehicle = sammi(r=des_ang, des_vel=des_vel, sample_time=sample_time, nu=initial_velocities, test_integrals=test_integrals)
    eta = initial_state
    nu = vehicle.nu
    u_actual = vehicle.u_actual
    if "eta" in tracking:
        # eta_hist = np.zeros((sample_count, 3))
        eta_hist = []
    if "nu" in tracking:
        nu_hist = np.zeros((sample_count, 3))
        nu_hist = []
    if "u_actual" in tracking:
        # u_actual_hist = np.zeros((sample_count, 2))
        u_actual_hist = []

    # for i in range(0, sample_count):
    i = 0
    while True:
        t = i * sample_time
        u_control = vehicle.update(eta, sample_time)
        if test_integrals:
            [nu, u_actual, integrals] = vehicle.dynamics(eta, nu, u_actual, u_control, sample_time)
        else:
            [nu, u_actual] = vehicle.dynamics(eta, nu, u_actual, u_control, sample_time)
        eta = attitudeEuler(eta, nu, sample_time)
        # print(eta[:2])

        if "eta" in tracking:
            # eta_hist[i] = eta
            eta_hist.append(copy.copy(eta))
        if "nu" in tracking:
            # nu_hist[i] = nu
            nu_hist.append(nu[:])
        if "u_actual" in tracking:
            # u_actual_hist[i] = u_actual
            u_actual_hist.append(u_actual[:])

        if end_condition[0] == "time":
            if t >= end_condition[1]:
                break
        elif end_condition[0] == "vel":
            vel = np.linalg.norm(nu[:2])
            if vel >= end_condition[1][0] - end_condition[1][1] and vel < end_condition[1][0] + end_condition[1][1]:
                break
        elif end_condition[0] == "ang":
            ang = np.rad2deg(eta[2])
            if abs(ang - end_condition[1][0]) <= end_condition[1][1]:
                break
        elif end_condition[0] == "vel ang":
            vel = np.linalg.norm(nu[:2])
            ang = np.rad2deg(eta[2])
            if vel >= end_condition[1][0] - end_condition[1][2] and vel < end_condition[1][0] + end_condition[1][2]:
                if abs(ang - end_condition[1][1]) <= end_condition[1][3]:
                    break

        i += 1
    
    return_data = []
    # print(eta_hist)
    if "eta" in tracking:
        return_data.append(eta_hist)
    if "nu" in tracking:
        return_data.append(nu_hist)
    if "u_actual" in tracking:
        return_data.append(u_actual_hist)

    if test_integrals:
        return_data.append([eta, nu, u_actual, integrals])
    else:
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
    eta[2] = eta[2] + sampleTime * nu[2]
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

def test_integrals():
    integral_dict = {}
    velocities = np.arange(0, 0.9, 0.01)
    for vel in velocities:
        data = simulate([0, 0, 0], [0, 0, 0], 0, vel, ("time", 30), 0.01, [], True)
        [eta, nu, u_actual, integral] = data[-1]
        # integral_dict[vel] = integral
        integral_dict[vel] = u_actual
        # print(vel, integral)
    print(integral_dict)
    

if __name__ == "__main__":
    # simulate([0, 0, 0], [0.505, 0, 0], 0, 0.5, ("time", 30), 0.01, [])
    try:
        timer = input("Do you want to time the simulation? (y/n): ")
        if timer == "n":
            des_vel = float(input("Enter a desired velocity: "))
            des_ang = float(input("Enter a desired angle: "))
            time = float(input("Enter a time for the simulation: "))
            end_condition = ("vel ang", [des_vel, des_ang])
        elif timer == "y":
            des_vel = 0.5
            des_ang = 0
            time = 5
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            # exit()
    except ValueError:
        print("Invalid input. Please enter valid numeric values for desired velocity and angle.")
        exit()
    
    # timer = "p"
    # initial_state = np.array([0, 0, 0, 0, 0, np.deg2rad(90)], float)
    initial_state = np.array([0, 0, 0], float)
    # initial_velocities = np.array([0, 0, 0, 0, 0, 0], float)
    initial_velocities = np.array([0, 0, 0], float)
    sample_time = 0.01
    tracking = ["eta"]


    if timer == "y":
        iterations = 100
        total_time = timeit.timeit("simulate(initial_state, initial_velocities, des_ang, des_vel, end_condition, sample_time, tracking)", setup="from __main__ import simulate, initial_state, initial_velocities, des_ang, des_vel, time, sample_time, tracking", number=iterations)
        print("Average time for {} iterations: ".format(iterations), total_time/iterations)
        print("Average time per simulation second: ", (total_time/iterations)/time)
    elif timer == "p":
        iterations = 1000
        for i in range(iterations):
            data = simulate(initial_state, initial_velocities, des_ang, des_vel, end_condition, sample_time, tracking)
            # print(initial_state)
        # print(data)
    else:
        des_vel = 0.5
        des_ang = 45
        # end_condition = ("vel ang", [des_vel, des_ang])
        # end_condition = ("time", 10)
        # end_condition = ("vel", [des_vel, 0.001])
        # end_condition = ("ang", [des_ang, 0.5])
        end_condition = ("vel ang", [des_vel, des_ang, 0.01, 0.5])
        data = simulate(initial_state, initial_velocities, des_ang, des_vel, end_condition, sample_time, tracking)
        [eta, nu, u_actual] = data[-1]
        print("Simulation complete", eta, nu, u_actual)

    # Plot positions
    if "eta" in tracking:
        eta_hist = np.array(data[0])
        plt.figure()
        plt.plot(eta_hist[:, 1], eta_hist[:, 0])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Vehicle Positions')
        plt.axis('equal')
        # plt.grid(True)
        plt.show()


