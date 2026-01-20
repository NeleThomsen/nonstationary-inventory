from typing import List, Tuple, Optional
import numpy as np


class Environment:
    """
    Environment class, defines the problem instance
    """
    def __init__(self, model: str, K: int, gamma: float | int, bslevels, 
                 h: float, b: float, d_dists: list, L: int=0):
        self.model = model
        self.K = K
        self.gamma = gamma
        self.bslevels = bslevels
        self.h = h
        self.b = b
        self.L = L
        self.onHandInventory = 0.0  # inventory after demand, before replenishment
        self.inTransit = np.zeros(L)
        self.sumInventory = 0.0
        
        self.d_dists = d_dists
        self.d_dist = d_dists[0]
        self.S = len(d_dists)
        self.num_current_state = 0
        self.exp_costs_list, self.exp_pseudo_costs_list = map(list, zip(*(d_dist.get_exp_cost(self) for d_dist in self.d_dists)))
        self.exp_costs = self.exp_costs_list[0]
        self.exp_pseudo_costs = self.exp_pseudo_costs_list[0]
        self.best_arms = [np.argmin(exp_pseudo_costs) for exp_pseudo_costs in self.exp_pseudo_costs_list]
        self.best_arm = self.best_arms[0]


    def __repr__(self) -> str:
        return f"Environment({self.model}, K={self.K}, d_dist={self.d_dist.__repr__}, h={self.h}, b={self.b})"

    def level_to_arm(self, level: float) -> int:
        """
        Map a continuous base-stock level tau in [0, U] to nearest discrete action index in {0, ..., K-1}.
        """
        level = min(max(0.0, float(level)), self.bslevels[self.K-1])
        arm = int(np.argmin(np.abs(np.asarray(self.bslevels) - level)))
        return arm

    def get_cost(self, arm) -> Tuple[float, float, float]:
        """
        Compute pseudo-cost, true cost, and sales for a given action (chosen inventory level ∈ {0, ..., K-1}).
        """
        demand = self.d_dist.sample()
        bslevel = self.bslevels[arm]
        
        if self.L:
            current_order = max(bslevel - self.sumInventory, 0.0)
            inventoryBeforeDemand = self.onHandInventory + self.inTransit[0]
            self.inTransit[0] = current_order
            self.inTransit = np.roll(self.inTransit, shift=-1)
        else:
            inventoryBeforeDemand = max(self.onHandInventory, bslevel)

        if self.model == "lost_sales":
            self.onHandInventory = max(0.0, inventoryBeforeDemand - demand)
        else:
            self.onHandInventory = inventoryBeforeDemand - demand

        self.sumInventory = (self.onHandInventory + sum(self.inTransit) if self.L else self.onHandInventory)

        sales = min(inventoryBeforeDemand, demand)
        pseudo_cost = self.h * (inventoryBeforeDemand - sales) - self.b * sales
        cost = pseudo_cost + self.b * demand

        if self.model == "lost_sales":
            return pseudo_cost, cost, sales
        else:
            return pseudo_cost, cost, demand

    def change(self):
        """
        Perform change in the demand distribution.
        """
        self.num_current_state += 1
        if self.num_current_state < self.S:
            self.d_dist = self.d_dists[self.num_current_state]
        else:
            assert f"Number of stationary segments {self.num_current_state} exceeds S={self.S}."
        self.exp_costs = self.exp_costs_list[self.num_current_state]
        self.exp_pseudo_costs = self.exp_pseudo_costs_list[self.num_current_state]
        self.best_arm = self.best_arms[self.num_current_state]