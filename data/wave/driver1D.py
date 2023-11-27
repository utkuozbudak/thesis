import numpy as np
import FiniteDifferenceNumpy as FiniteDifference
import matplotlib.pyplot as plt
import time

Lx = 1

Nx = 100 
dx = Lx / Nx
dt = 1e-3 # Time step size
N = 400 # Number of time steps

x = np.linspace(0-dx, Lx+dx, Nx + 3) # with ghost cells
t = np.linspace(0, (N-1) * dt, N)
x_, t_ = np.meshgrid(x, t, indexing='ij')

gamma = x * 0 + 1.
# index = (x > 0.75)
# gamma[index] = 1e-1
def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


rho = 2.7e0
c = 10

def diracx(x, i):
    x = x*0
    x[i] = 1
    return x

def generateSineBurst( frequency, cycles ):
    omega = frequency * 2 * np.pi
    # return lambda t : 1e19 * ((t <= cycles/frequency ) & (t > 0)) * np.sin( omega * t ) * (np.sin( omega * t / 2 / cycles))**2
    return lambda t : 1e4 * ((t <= cycles/frequency ) & (t > 0)) * np.sin( omega * t ) * (np.sin( omega * t / 2 / cycles))**2

frequency = 30
cycles = 3

source = generateSineBurst(frequency, cycles) 
f_source = lambda x, t, i : diracx(x, i) * source(t)
f = f_source(x_, t_, 1)[1:-1,:]

u0 = x*0
u1 = x*0

start = time.perf_counter()
U = FiniteDifference.finiteDifference1D(u0.copy(), u1.copy(), f, c, rho, gamma, dx, Nx, dt, N)
end = time.perf_counter()
print("Elapased time: {:2f} ms".format((end-start)*1000))

# fig, ax = plt.subplots()
# cp = ax.pcolormesh(x_, t_, U[:,1:], cmap=plt.cm.seismic)
# fig.colorbar(cp)
# plt.show()


fig,ax = plt.subplots(1,4,figsize=(12,3))
cp = ax[0].pcolormesh(x_, t_, U[:,1:], cmap=plt.cm.seismic)
fig.colorbar(cp)
ax[0].set_title("wave")
# plt.show()

ax[1].plot(t,f[0,:])
ax[1].set_title("forcing")

ax[2].plot(x,gamma)
ax[2].set_title("gamma")

ax[3].plot(U[85,:])
ax[3].set_title("Displacement plot")

plt.tight_layout()
plt.savefig("out.png")
# plt.show()

np.save("U.npy",U[:,1:])
np.save("gamma.npy",gamma[:])
np.save("f.npy",f[:,:])
print(U[:,1:].shape)
print(gamma[:].shape)
print(f[:,:].shape)