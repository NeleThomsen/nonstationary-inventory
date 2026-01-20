from algorithms.algoClass import BaseAlgo
from math import log, sqrt, ceil
from random import random
import numpy as np


class NSIC(BaseAlgo):
    def __init__(self, K: int, T: int, L: int = 0, U: float = 1.0, model: str = "lost_sales",
                 lipschitzConst: float = 1.0, deltaProb: float = 0.05, verbose: bool = True,
                 constChangeCheck: float = 1.0, constEvictionCheck: float = 1.0):
        """
        :param K: number of discrete actions 0, ..., K-1 (indices in list bslevels of Environment)
        :param T: time horizon
        :param L: deterministic lead time
        :param U: highest base-stock level
        :param model: lost_sales or backlogging
        :param lipschitzConst: Lipschitz constant max(h,b)
        :param deltaProb: probability
        :param verbose: -
        :param costChangeCheck: constant
        :param constEvictionCheck: constant
        """
        self.K = K
        self.T = T
        self.L = L
        self.U = U
        self.model = model
        self.lipschitzConst = lipschitzConst
        self.deltaProb = deltaProb
        self.verbose = verbose
        self.constChangeCheck = constChangeCheck
        self.constEvictionCheck = constEvictionCheck

        if self.model == "lost_sales":
            self.H = 72.0 * (self.L + 3) * U * lipschitzConst
            self.const_b_st = self.H * np.sqrt(2 * np.log(2/self.deltaProb))
        else:
            self.H = 2 * np.sqrt(2) * np.sqrt((self.L+1)*(self.L+pow(50,2)*(4*self.L+5)))
            self.const_b_st = self.H * np.sqrt(2 * np.log(4*(self.L+1)/self.deltaProb))

        self.active_set = list(range(K))
        self.tau_vk = max(self.active_set)
        self.v = 1  # counter for episodes
        self.t = 1
        self.t_v = 1  # starting time of current episode
        self.alpha_vk = 1  # starting time of current epoch in current episode
        if self.model == "lost_sales" and not self.L:
            self.samplingObligations = 0
            self.deltaTilde = [self.lipschitzConst * U for _ in range(K)]
            self.muTilde = [self.lipschitzConst * self.L * U for _ in range(K)]

        self.delta_check_timesteps = 50
        self.freq_change_checks = 50
        self.freq_eviction_checks = 20
        self.actionHistory = []
        self.pseudoCostHistory = np.full((K, T + 2), np.nan, dtype=float)
        self.invStates = np.zeros((K, L + 1), dtype=float)

    def __repr__(self):
        return f"NSIC(T={self.T}, L={self.L})"

    def clear(self):
        self.active_set = list(range(self.K))
        self.tau_vk = max(self.active_set)
        if self.model == "lost_sales":
            self.H = 72.0 * (self.L + 3) * self.lipschitzConst * self.U
            self.const_b_st = self.H * np.sqrt(2 * np.log(2/self.deltaProb))
        else:
            self.H = 2 * np.sqrt(2) * np.sqrt((self.L+1)*(self.L+pow(50,2)*(4*self.L+5)))
            self.const_b_st = self.H * np.sqrt(2 * np.log(4*(self.L+1)/self.deltaProb))
        self.t = 1
        self.v = 1
        self.t_v = 1
        self.alpha_vk = 1
        if self.model == "lost_sales" and not self.L:
            self.samplingObligations = 0
            self.deltaTilde = [self.lipschitzConst * self.U for _ in range(self.K)]
            self.muTilde = [self.lipschitzConst * self.L * self.U for _ in range(self.K)]
        self.actionHistory = []
        self.pseudoCostHistory = np.full((self.K, self.T + 2), np.nan, dtype=float)
        self.invStates = np.zeros((self.K, self.L + 1), dtype=float)

    def start_new_episode(self):
        self.active_set = list(range(self.K))
        self.v += 1
        self.t_v = self.t
        if self.model == "lost_sales" and not self.L:
            self.deltaTilde = [self.lipschitzConst * self.U for _ in range(self.K)]
            self.muTilde = [self.lipschitzConst * self.L * self.U for _ in range(self.K)]
        self.actionHistory = []
        self.pseudoCostHistory[:] = np.nan
        self.invStates[:] = 0.0
        self.start_new_epoch()

    def start_new_epoch(self):
        self.tau_vk = max(self.active_set)
        self.alpha_vk = self.t
        if self.L and self.model == "lost_sales":
            self.invStates[:] = 0.0

    def n_s_t(self, arm: int, s: int, t: int) -> int:
        assert s <= t, f"Variable n_s_t only exists for s <= t, but s={s} and t={t} for action a={arm}"
        values = self.pseudoCostHistory[arm, s:t+1]
        return np.count_nonzero(~np.isnan(values))

    def mu_hat_s_t(self, arm: int, s: int, t: int) -> float:
        assert s <= t, f"Variable mu_hat_s_t only exists for s <= t, but s={s} and t={t} for action a = {arm}"
        slice_ = self.pseudoCostHistory[arm, int(s):int(t)+1]
        if np.all(np.isnan(slice_)):
            return 0.0
        return float(np.nanmean(slice_))

    def check_changes(self) -> bool:
        if self.verbose:
            print("\nChecking actions in active set for changes.")
        lost_sales = (self.model == "lost_sales")

        s_init = self.alpha_vk if lost_sales and self.L else self.t_v
        highest_arm_observed = self.tau_vk if lost_sales else self.K - 1
        factor = self.constChangeCheck * self.const_b_st

        # Precompute time grid
        S = list(range(s_init, self.t + 1, self.delta_check_timesteps))

        for arm in range(0, highest_arm_observed + 1):
            n_cache = {}
            mu_cache = {}
            for index_step, s1 in enumerate(S):
                for s2 in S[index_step+1:]:
                    key = (s1, s2)
                    n_val = self.n_s_t(arm, s1, s2)
                    n_cache[key] = n_val
                    if n_val >= 5:
                        mu_cache[key] = self.mu_hat_s_t(arm, s1, s2)

            for index_step, s1 in enumerate(S):
                for s2 in S[index_step+1:]:
                    key12 = (s1, s2)
                    n_s1_s2_a = n_cache[key12]
                    if n_s1_s2_a < 5:
                        continue

                    mu_hat_s1_s2_a = mu_cache[key12]
                    inv_sqrt_n12 = 1.0 / np.sqrt(n_s1_s2_a)

                    for s in S[index_step+1:]:
                        if s < s2:
                            continue
                        n_s_t_a = self.n_s_t(arm, s, self.t)
                        if n_s_t_a < 5:
                            break

                        mu_hat_s_t_a = self.mu_hat_s_t(arm, s, self.t)

                        diff = abs(mu_hat_s1_s2_a - mu_hat_s_t_a)
                        conf = factor * (inv_sqrt_n12 + 1.0 / np.sqrt(n_s_t_a))

                        if diff > conf:
                            if self.verbose:
                                print(f"\nNew episode: change in costs of action {arm}, "
                                    f"[s1, s2] = [{s1}, {s2}], [s, t] = [{s}, {self.t}]")
                            return True
        return False
    
    def check_changes_bad_arms(self) -> bool:
        if self.verbose:
            print("\nChecking eliminated actions for changes.")
        factor = self.constChangeCheck * self.const_b_st
        S = range(self.t_v, self.t + 1, self.delta_check_timesteps)

        for arm in range(self.tau_vk + 1, self.K):
            muTilde_arm = self.muTilde[arm]
            deltaTilde_arm = self.deltaTilde[arm]
            deltaTilde_four = deltaTilde_arm / 4.0

            for s in S:
                # n(s, t) is decreasing as s increases, hence once below threshold, break.
                n_s_t_a = self.n_s_t(arm, s, self.t)
                if n_s_t_a < 5:
                    break

                mu_hat_s_t_a = self.mu_hat_s_t(arm, s, self.t)
                diff = abs(mu_hat_s_t_a - muTilde_arm)

                conf = factor / np.sqrt(n_s_t_a)
                right_side = deltaTilde_four + conf

                if diff > right_side:
                    if self.verbose:
                        print(f"\nNew episode: change in costs of action {arm}, "
                            f"[s1, s2] = [s, t] = [{s}, {self.t}] ")
                    return True
        return False
    
    def check_eviction(self, gamma) -> bool:
        if not self.active_set:
            return False
        
        lost_sales = (self.model == "lost_sales")
        s_init = self.alpha_vk if lost_sales and self.L else self.t_v
        t = self.t

        active_set_updated = False
        arms_to_remove = set()

        if lost_sales and not self.L:
            factor = self.constEvictionCheck * 6 * self.const_b_st
        else:
            factor = self.constEvictionCheck * 4 * self.const_b_st

        for s in range(s_init, t, self.delta_check_timesteps):
            # Remove in each iteration over s the actions that satisfy elimination (and separation) condition
            active_indices = np.array(self.active_set, dtype=int)
            slice_idx = np.arange(s, t + 1)

            # Compute mean only for slices that have at least one valid entry
            values = self.pseudoCostHistory[active_indices[:, None], slice_idx]
            with np.errstate(all='ignore'):
                mu_hat = np.nanmean(values, axis=1)

            valid_mask = ~np.isnan(mu_hat)
            if not np.any(valid_mask):
                continue  # nothing to eliminate if all NaNs

            valid_indices = active_indices[valid_mask]  # all indices in active set for which there is at least one observation in [s,t]
            valid_mu = mu_hat[valid_mask]

            lowest_cost = np.min(valid_mu)
            diffs = valid_mu - lowest_cost

            n_s_t_a = np.sum(~np.isnan(values), axis=1)
            n_s_t_a = n_s_t_a[valid_mask]
            if np.all(n_s_t_a < 5):
                break  # early exit for increasing s
            conf_radii = factor / np.sqrt(n_s_t_a)
            eliminate_mask = diffs > conf_radii
            if not np.any(eliminate_mask):
                continue
            
            index_map = {arm: i for i, arm in enumerate(active_indices)}

            for arm, diff, conf, n in zip(
                valid_indices[eliminate_mask],
                diffs[eliminate_mask],
                conf_radii[eliminate_mask],
                n_s_t_a[eliminate_mask],):
                
                prev_arm = arm - 1
                if prev_arm in valid_indices:
                    prev_mu = valid_mu[np.where(valid_indices == prev_arm)[0][0]]
                else:
                    prev_mu = float("inf")

                separation_ok = (not self.L
                                or self.model == "backlog"
                                or (prev_mu - lowest_cost)
                                >= 2 * self.constEvictionCheck * self.const_b_st / np.sqrt(n)
                                + self.lipschitzConst * gamma)

                if separation_ok:
                    arms_to_remove.add(arm)

                    if lost_sales and not self.L:
                        self.deltaTilde[arm] = diff
                        self.muTilde[arm] = mu_hat[index_map[arm]]

                    if self.verbose:
                        print(f"\n--- Eliminating action {arm} from active set, [s, t] = [{s}, {t}] "
                              f"and abs diff={diff - conf}")

            # Update active set
            if arms_to_remove:
                active_set_updated = True
                self.active_set = [arm for arm in self.active_set if arm not in arms_to_remove]
                active_indices = np.asarray(self.active_set, dtype=int)

        return active_set_updated

    def updateAlgo(self, arm: int, cost: float, sales: float, environment):
        # under backlogging, sales is the true demand here
        if self.verbose:
            print(f"selected a={arm}/{self.K-1} in t={self.t}")

        self.pseudoCostHistory[arm, self.t] = cost
        lost_sales = (self.model == "lost_sales")

        if lost_sales and not self.L and self.tau_vk < self.K - 1:
            # add sampling obligations for tau=U
            max_i = - int(log(environment.gamma) / log(2))
            for index in range(1, max_i + 1):
                if random() <= pow(2, -index) * sqrt(self.v/(self.U * self.T * log(2*pow(self.T,2)*self.U/(self.deltaProb*environment.gamma)))):
                    self.samplingObligations += ceil(pow(2, 2*index + 1) * log(2*pow(self.T,2)*self.U/(self.deltaProb*environment.gamma)))

        # update counterfactual inventory positions and cost histories
        highest_arm_observed = arm if lost_sales else self.K-1        
        m = highest_arm_observed + 1

        cf_bslevels = np.asarray(environment.bslevels[:m], dtype=float)
        inv = self.invStates[:m] # shape (m, L+1)

        if self.L:
            cf_order = np.maximum(cf_bslevels - np.sum(inv, axis=1), 0.0)
            cf_inventoryBeforeDemand = inv[:,0] + inv[:,1]     
            cf_sales = np.minimum(cf_inventoryBeforeDemand, sales)
            on_hand_after = cf_inventoryBeforeDemand - (cf_sales if lost_sales else sales)
            cf_inTransit = np.concatenate((inv[:,2:self.L+1], cf_order[:,None]), axis=1)
            self.invStates[:m] = np.concatenate((on_hand_after[:,None], cf_inTransit), axis=1)
        else:
            cf_inventoryBeforeDemand = np.maximum(inv[:, 0], cf_bslevels)
            cf_sales = np.minimum(cf_inventoryBeforeDemand, sales)           
            self.invStates[:m, 0] = cf_inventoryBeforeDemand - (cf_sales if lost_sales else sales)

        self.pseudoCostHistory[:m, self.t] = environment.h * (cf_inventoryBeforeDemand - cf_sales) - environment.b * cf_sales

        self.t += 1
        # Change point detection
        if self.t % self.freq_change_checks == 0:
            if self.check_changes() or (lost_sales and not self.L and self.check_changes_bad_arms()):
                self.start_new_episode()
                return

        # Active set elimination
        if self.t % self.freq_eviction_checks == 0:
            active_set_updated = self.check_eviction(environment.gamma)
            if active_set_updated:
                self.start_new_epoch()


    def selectAction(self, environment) -> int:
        if self.model == "lost_sales" and not self.L and self.samplingObligations > 0:
            a_t = self.K-1
            self.samplingObligations -= 1
        else:
            a_t = self.tau_vk

        self.actionHistory.append(a_t)
        return a_t  # returns arm (index in bslevels), not base-stock level
