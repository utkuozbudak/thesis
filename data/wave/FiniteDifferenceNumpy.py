import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

"""
Finite Difference Solver for the 2D wave equation
Adapted from: http://hplgit.github.io/INF5620/doc/notes/wave-sphinx/main_wave.html
"""

def finiteDifferenceStep1D(u0, u1, f, c, rho, gamma, dx, Nx, dt):
    u2 = np.zeros((Nx + 3))
    # central difference scheme with harmonic mean instead for gamma
    u2[1:-1] = -u0[1:-1] + 2 * u1[1:-1] + \
               (dt * c / dx)**2 / gamma[1:-1] * ((0.5 / gamma[1:-1] + 0.5 / gamma[2:])**(-1) * (u1[2:] - u1[1:-1]) - \
                                                 (0.5 / gamma[:-2] + 0.5 / gamma[1:-1])**(-1) * (u1[1:-1] - u1[:-2])) + \
                dt**2 / (rho * gamma[1:-1]) * f[:]
    
    # update of ghost cells (corners don't need to be updated, as they are not used)
    u2[0] = u2[2]
    u2[-1] = u2[-3]
    
    return u2

def finiteDifference1D(u0, u1, f, c, rho, gamma, dx, Nx, dt, N):
    
    Cx = c*dt/dx
    if Cx**2 > 1:
        print('Warning: Courant number is larger than 1')
    
    U = np.zeros((Nx + 3, N + 1))
    U[:,0] = u1
    
    for timestep in range(N):
        u2 = finiteDifferenceStep1D(u0, u1, f[:, timestep], c, rho, gamma, dx, Nx, dt)
        u0[:], u1[:] = u1, u2
        
        U[:,timestep+1] = u2
        
    return U
