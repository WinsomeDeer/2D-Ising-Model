import numpy as np
import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt

#libc.stdlib cimport rand, srand, RAND_MAX, abs
#from libc.math cimport sqrt, pow, exp
#from libc.time cimport time

# ----------------------------- TBF ------------------------------------------


# Class for Ising model.
class Ising_Model:
    # Constructor.
    def __init__(self, beta, J, m, n):
        self.beta = beta
        self.m = m
        self.n = n
        self.J = J
        self.lattice = self.build_lattice()
    # function to build a lattice of random values.
    def build_lattice(self):
        lattice = np.zeros(shape = (self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                lattice[i, j] = np.random.choice([-1,1])
        return lattice
    # function to calculate the nearest neighbour (nn) energy.
    def nn_energy(self, m, n):
        nn = 0
        for i in range(m-1, m+1):
            for j in range(n-1, n+1):
                if i == m and j == n:
                    continue
                else:
                    nn += self.lattice[i % self.m, j % self.n]
        return nn
    # function to update the lattice.
    def itr(self, frame_num, img):
        for i in range(self.m):
            for j in range(self.n):
                nn = self.nn_energy(i, j)
                Q = 2*nn*self.lattice[i, j]
                if Q < 0:
                    self.lattice[i, j] *= 1
                elif np.random.rand() < np.exp(-Q*self.beta):
                    self.lattice[i, j] *= 1
        print(np.sum(self.lattice))
        img.set_data(self.lattice)
        return img
# Main function.
def main():
    a = Ising_Model(0.4, 1, 200, 200)
    fig, ax = plt.subplots()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    img = ax.imshow(a.build_lattice(), interpolation = 'nearest')
    ani = animation.FuncAnimation(fig, a.itr, fargs=(img,), frames = 10, interval = 50, save_count = 50)
    plt.show()
if __name__ == '__main__':
    main()
