from abc import ABC, abstractmethod
from MCSimPython.thrust_allocation.thruster import Thruster
from itertools import repeat
import numpy as np

DOFS = 3

class AllocationError(Exception):
    """
    AllocationError class
    """

class AllocatorCSAD(ABC):
    """
    Abstract base class for allocation problem
    formulations. Spesific for CSAD.
    """

    def __init__(self):
        self._thrusters = []

    @property
    def n_thrusters(self):
        """
        Number of thrusters assigned to this
        allocation problem.
        """
        return len(self._thrusters)

    def add_thruster(self, thruster):
        """
        Add a thruster to the allocation problem.
        """
        if isinstance(thruster, Thruster):
            self._thrusters.append(thruster)
        else:
            raise TypeError("Thruster is not of proper type!")
    

    @abstractmethod
    def allocation_problem(self):
        """
        Assemble allocation problem into matrix form
        """


    @abstractmethod
    def allocate(self, tau_d):
        """
        Allocate global thrust vector to available thrusters. 
        """
    

class pseudo_inverse_allocator(AllocatorCSAD):

    @property
    def n_problem(self):
        """
        Number of unknown variables to be allocated
        in the original problem.
        """
        return 2 * self.n_thrusters

    def allocation_problem(self):

        """
        Assemble extended allocation problem into matrix form.
        """

        T_e = np.zeros((DOFS, self.n_problem))
        T_e[0, ::2] = 1
        T_e[1, 1::2] = 1
        T_e[2, ::2] = [-thruster.pos_y for thruster in self._thrusters]
        T_e[2, 1::2] = [thruster.pos_x for thruster in self._thrusters]
        
        K_lst = [thruster._K for thruster in self._thrusters]

        K_vec = [x for item in K_lst for x in repeat(item, 2)]

        K_e = np.diag(K_vec)

        return T_e, K_e
    

    def allocate(self, tau_d):

        """
        Allocate global thrust vector to available thrusters. 
        """

        if self.n_thrusters == 0:
            raise AllocationError(
                """At least one thruster must be added
                    to the
                    allocator-object before attempting an allocation!"""
            )
        
        T_e, K_e = self.allocation_problem()

        T_e_pseudo = np.transpose(T_e) @ (np.linalg.inv(T_e @ np.transpose(T_e)))

        u_e = np.linalg.inv(K_e) @ T_e_pseudo @ tau_d

        u = np.zeros(self.n_thrusters)
        alpha = np.zeros(self.n_thrusters)

        for i in range(self.n_thrusters):
            u[i] = np.sqrt(u_e[i*2]**2 + u_e[i*2+1]**2)
            alpha[i] = np.arctan2(u_e[i*2+1], u_e[i*2])
        
        return u, alpha


class fixed_angle_allocator(AllocatorCSAD):

    theta = np.deg2rad(30)

    alpha = np.array([np.pi,
             np.pi/2 + theta,
             3 * np.pi/2 - theta,
             - np.pi,
             - np.pi/2 + theta,
             np.pi/2 - theta])

    def allocation_problem(self):
        """
        Assemble extended allocation problem into matrix form
        """

        K_lst = [thruster._K for thruster in self._thrusters]

        lx = [thruster.pos_x for thruster in self._thrusters]
        ly = [thruster.pos_y for thruster in self._thrusters]

        K = np.diag(K_lst)
        
        T = np.array([
            np.cos(self.alpha),
            np.sin(self.alpha),
            lx * np.sin(self.alpha) - ly * np.cos(self.alpha)])

        return T, K

    def allocate(self, tau_d):
        """
        Allocate global thrust vector to available thrusters. 
        """

        if self.n_thrusters == 0:
            raise AllocationError(
                """At least one thruster must be added
                    to the
                    allocator-object before attempting an allocation!"""
            )
        
        T, K = self.allocation_problem()

        B = T @ K

        B_inv = np.transpose(B) @ (np.linalg.inv(B @ np.transpose(B)))

        u = B_inv @ tau_d

        alpha = self.alpha
        
        return u, alpha
