import numpy as np
import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt

#libc.stdlib cimport rand, srand, RAND_MAX, abs
#from libc.math cimport sqrt, pow, exp
#from libc.time cimport time

# Boltzmans constant (k) and electron magnetic moment (mu).
k = 1.38 * 10 ** -27
mu = -9.2847647043 * 10 ** -24

# Class for Ising model.
class Ising_Model:
    # Constructor.
    def __init__(self, beta, m, n):
        self.beta = beta
        self.m = m
        self.n = n
        self.lattice = self.build_lattice()
    # Function to build a lattice of random values.
    def build_lattice(self):
        lattice = np.zeros(shape = (self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                lattice[i, j] = np.random.choice([-1,1])
        return lattice
    # Magnetization function
    def magnetization(self):
        S = 0
        for i in range(self.m):
            for j in range(self.n):
                S += self.lattice[i, j]
        return  S
    # Magnetic Suspectibility Function
    def magnetic_susceptibility(self):
        M = 0
        M_2 = 0
        for i in range(self.m):
            for j in range(self.n):
                M += self.lattice[i, j]
                M_2 += self.lattice[i, j] ** 2
        return (M_2 - M ** 2) * (self.beta / k)
    # Internal Energy function
    def internal_energy(self):
        U = 0 
        for i in range(self.m):
            for j in range(self.n):
                U += self.nn_energy(i, j)
        return U
    # Specific Heat function
    def specific_heat(self):
        e = 0 
        E = 0
        E_2 = 0
        for i in range(self.m):
            for j in range(self.n):
                e = self.nn_energy(i, j)
                E += e
                E_2 += e**2
        U = (1./self.m * self.n) * E
        U_2 = (1./self.m * self.n) * E_2
        return (U_2 - U ** 2) * (1/k * self.beta)
    # function to calculate the nearest neighbour (nn) energy.
    def nn_energy(self, i, j):
        return (self.lattice[(i+1)%self.m, j] + self.lattice[(i-1)%self.m, j] + self.lattice[i, (j+1)%self.n] + self.lattice[i, (j-1)%self.n])               
    # function to update the lattice.
    def itr(self, frame_num, img):
        E, M, M_abs = 0, 0, 0
        E_tot, E_totsq, M_tot, M_totsq, M_totabs = 0, 0, 0, 0, 0
        for i in range(10):
            for j in range(self.m * self.n):
                a = np.random.randint(0, self.m)
                b = np.random.randint(0, self.n)
                nn = self.nn_energy(a, b)
                Q = 2*nn*self.lattice[a, b]
                if Q < 0:
                    self.lattice[a, b] *= -1
                    E += Q
                    M += 2 * self.lattice[a, b]
                    M_abs += abs(self.lattice[a, b])
                elif np.random.rand() < np.exp(-Q*self.beta):
                    self.lattice[a, b] *= -1
                    E += Q
                    M += 2 * self.lattice[a, b]
                    M_abs += abs(self.lattice[a, b])
            E_tot += E/2.0
            E_totsq += (E ** 2)/4.0
            M_tot += M
            M_totsq += (M ** 2)
            M_totabs += abs(M)
        print(M_totabs / (self.m * self.n))
        img.set_data(self.lattice)
        return img

# Main function.
def main():
    a = Ising_Model(1, 200, 200)
    fig, ax = plt.subplots()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    img = ax.imshow(a.build_lattice(), interpolation = 'nearest')
    ani = animation.FuncAnimation(fig, a.itr, fargs = (img,), frames = 10, interval = 50, save_count = 50)
    plt.show()

if __name__ == '__main__':
    main()
