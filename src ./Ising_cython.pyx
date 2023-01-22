import numpy as np
cimport numpy as np
import random
import matplotlib.pyplot as plt
import cython
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport sqrt, pow, exp
from libc.time cimport time
srand(time(NULL))
# Class to initialise the grid.
cdef int n = 50
cdef double k = 1.38 * 10 ** -27
# Lattice Class.
class Ising_lattice:
    # Constructor
    def __init__(self, int n, int temperature, int J):
        self.n = n
        self.T = temperature
        self.J = J
        self.lattice = self.build_lattice()
    def np.ndarray build_lattice(self):
        cdef int i, j
        cdef np.ndarray[np.int32_t, ndim = 2] lattice
        for i in range(n):
            for j in range(n):
                lattice[i,j] = rand()/ RAND_MAX
                if lattice[i,j] < 0.5:
                    lattice[i,j] = 1
                else:
                    lattice[i,j] = -1
        return lattice
    # Property of internal energy.
    @property
    def Internal_energy(self):
        cdef int i, j, e = 0, E = 0, E_2 = 0, U, U_2
        # Calculate the energy for each cell.
        for i in range(self.n):
            for j in range(self.n):
                e = self.energy(i, j, self.J, self.T)
                E += e
                E_2 += e**2
        U = (1/self.n**2) * E
        U_2 = (1/self.n**2) * E_2
        return U, U_2
    # Heat capacity property.
    @property
    def heat_capacity(self):
        cdef int U, U_2
        U, U_2 = self.Internal_energy
        return U_2 - U
    # Magnetisation property.
    @property
    def magnetisation(self):
        return np.abs(np.sum(self.lattice)/self.n**2)
    # Function to run one iteration.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def one_itr(self, int[:,::1] lattice):
        cdef int i, j 
        for i in range(self.n):
            for j in range(self.n):
                S = self.lattice[i,j]
                nn = (self.lattice[(i+1)%self.n, j] + self.lattice[(i-1)%self.n, j]
                         + self.lattice[i, (j+1)%self.n] + self.lattice[i, (j-1)%self.n])               
                Q = 2*S*nn
                if Q < 0:
                    S *= -1
                elif rand() < exp(-Q*1.0/self.T) * RAND_MAX:
                    S *= -1
                self.lattice[i,j] = S
        return self.lattice

