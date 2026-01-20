import os
import csv
import yaml
import datetime
import random
import optuna
from multiprocessing import Pool, cpu_count
import copy
from helpers.helpers import get_env_from_config, get_best_fixed_arm_hindsight
from environments.envClass import *
from algorithms.algorithm import NSIC

def runAlgorithm(algorithm,
                 environment: Environment,
                 T: int,
                 changes=None):
    """
    :param algorithm: algorithm
    :param environment: Environment
    :param T: length of time horizon (number of time steps)
    :param changes: list of change points
    :return: regret, true_cost, num_neg_orders, best_action_history
    """
    dynamic_regret = []
    best_arm_hindsight, exp_cost_best_arm_hindsight = get_best_fixed_arm_hindsight(environment, T, changes)
    static_regret = []
    true_cost = []
    expected_cost = []
    sum_min_exp_cost = 0.0
    min_exp_cost = environment.exp_costs[environment.best_arm]

    print(f"\nUsing {algorithm.__repr__()} with S={environment.S}.")
    while algorithm.t <= T:
        if algorithm.t in changes:  # change in demand distribution
            environment.change()
            plt_ex_cost = environment.exp_costs
            min_exp_cost = environment.exp_costs[environment.best_arm]
            if algorithm.verbose:
                print(f"Change occurred in t={algorithm.t}. New best arm: {environment.best_arm} with level {environment.bslevels[environment.best_arm]}")

        arm = algorithm.selectAction(environment)
        pseudo_cost, cost, sales = environment.get_cost(arm)  # arm is index of environment.bslevels, and bslevel if not algorithm.discrete_A # under backlogging, sales is demand
        true_cost.append(cost)
        expected_cost.append(environment.exp_costs[arm])
        dynamic_regret.append(environment.exp_costs[arm] - min_exp_cost)
        static_regret.append(environment.exp_costs[arm] - exp_cost_best_arm_hindsight)
        sum_min_exp_cost += min_exp_cost

        algorithm.updateAlgo(arm, pseudo_cost, sales, environment)  # under backlogging, sales is the demand here
        
    return dynamic_regret, static_regret, true_cost, expected_cost, sum_min_exp_cost

_global_algorithm = None

def _init_worker(algorithm):
    """Each worker gets its own independent algorithm instance."""
    global _global_algorithm
    _global_algorithm = copy.deepcopy(algorithm)


def _run_single(repetition, config, T):
    """
    Worker that initializes its own environment and algorithm copy, then runs the simulation.
    Executed in a separate process by multiprocessing.Pool.
    """
    global _global_algorithm
    
    if _global_algorithm is None:
        raise RuntimeError("Algorithm not initialized in worker")
    algorithm = _global_algorithm   # worker-local instance

    # create environment and U for this seed
    environment, U, _ = get_env_from_config(config=config, seed=config['seed'] + repetition)
    # ensure algorithm instance is prepared for this environment
    if hasattr(environment, "K"):
        algorithm.K = environment.K
    if hasattr(environment, "U"):
        algorithm.U = U
    if hasattr(algorithm, "clear"):
        algorithm.clear()

    # ordered list of change points
    points = range(T)
    if len(environment.d_dists) > 1:
        changePoints = sorted(random.sample(points, len(environment.d_dists)-1))
        changePoints = [points[index] for index in changePoints]
    else:
        changePoints = []

    start_time = datetime.datetime.now()
    regret, static_regret, true_cost, expected_cost, cum_min_exp_cost = runAlgorithm(algorithm=algorithm, environment=environment, 
                                                                                     T=T, changes=changePoints)
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Completed repetition {repetition + 1} in {duration:.2f}s with average regret {np.mean(regret):.4f}.")
    return (repetition, regret, static_regret, true_cost, expected_cost, cum_min_exp_cost, duration)


def main(algorithm,
         environment: Environment,
         T: int,
         num_repetitions: int = 1,
         config: dict = {}):
    """
    :param algorithm: algorithm
    :param algo_name: name of the algorithm (str)
    :param environment: environment instance (Environment)
    :param T: size of time horizon (int)
    :param num_repetitions: number of repetitions to run (int)
    :param config: config file containing input data
    :return: performance_metrics
    """
    assert algorithm is not None, "No algorithm specified."
    performance_metrics = {}
    avg_runtime = 0.0
    if num_repetitions == 1:
        # single-run (sequential) behaviour unchanged
        # ordered list of change points
        points = range(T)
        changePoints = sorted(random.sample(points, len(environment.d_dists)-1)) if len(environment.d_dists) > 1 else []
        changePoints = [points[index] for index in changePoints] if changePoints else []

        start_time = datetime.datetime.now()
        regret, static_regret, true_cost, expected_cost, cum_min_exp_cost = runAlgorithm(algorithm=algorithm, environment=environment, 
                                                                                         T=T, changes=changePoints)
        timeDelta = datetime.datetime.now() - start_time
        avg_runtime += timeDelta.total_seconds()
        print(f"Run 1/{num_repetitions} of {algorithm.__repr__()} completed in {timeDelta.total_seconds():.4f}s with average regret {np.mean(regret)}.")

        performance_metrics[0] = {
            'R': np.cumsum(regret, axis=0, dtype=float),
            'SR': np.cumsum(static_regret, axis=0, dtype=float),
            'C': np.cumsum(true_cost, axis=0, dtype=float),
            'EC': np.cumsum(expected_cost, axis=0, dtype=float),
            'RR': np.sum(regret)/cum_min_exp_cost * 100
        }
    else:
        # parallel execution of repetitions
        processes = min([cpu_count() - 2, num_repetitions, config['num_cpu_cores']])
        with Pool(
            processes=processes,
            initializer=_init_worker,
            initargs=(algorithm,)
        ) as pool:
            tasks = [(r, config, T) for r in range(num_repetitions)]
            results = pool.starmap(_run_single, tasks)

        # aggregate results
        for (repetition, regret, static_regret, true_cost, expected_cost, cum_min_exp_cost, duration) in results:
            avg_runtime += duration
            performance_metrics[repetition] = {
                'R': np.cumsum(regret, axis=0, dtype=float),
                'SR': np.cumsum(static_regret, axis=0, dtype=float),
                'C': np.cumsum(true_cost, axis=0, dtype=float),
                'EC': np.cumsum(expected_cost, axis=0, dtype=float),
                'RR': np.sum(regret)/cum_min_exp_cost * 100
            }
            print(f"Run {repetition + 1}/{num_repetitions} of {algorithm.__repr__()} completed in {duration:.2f}s with average regret {np.mean(regret):.4f}.")
    if not opt_params:
        # Write to CSV
        output_dir = os.path.join("logs", f"simulations_{environment.model}/{algorithm.__repr__()}")
        output_file = os.path.join(output_dir, f"L{environment.L}_S{len(environment.d_dists)}_{environment.d_dists[0]}.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["repetition", "R", "SR", "C", "EC", "RR"])

            for rep, met in performance_metrics.items():
                writer.writerow([
                    rep,
                    met["R"][-1],
                    met["SR"][-1],
                    met["C"][-1],
                    met["EC"][-1],
                    met["RR"]
                ])

        # Write regret time series to CSV
        output_file = os.path.join(output_dir, f"Regret_L{environment.L}_S{len(environment.d_dists)}_{environment.d_dists[0]}.csv")
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for rep in performance_metrics:
                writer.writerow(performance_metrics[rep]['R'])

    print(f"{algorithm.__repr__()}: average runtime per run: {avg_runtime / max(1, num_repetitions):.4f} ")
    return performance_metrics


def objective(trial: optuna.Trial, env) -> float:
    """ objective function for hyperparameter optimization """
    if "NSIC" in config['algorithm']:
        const1 = trial.suggest_float("const1", 1e-8, 1e-2, log=True)
        const2 = trial.suggest_float("const2", 1e-8, 1e-2, log=True)
        print(f"Trial {trial.number} - Params: const1={const1}, const2={const2}")
        algorithm = NSIC(K=env.K, T=T_horizon, L=env.L, U=U,
                                      model=env.model, lipschitzConst=lipschitzFactor, deltaProb=config['delta_prob'],
                                      verbose=verbose, constChangeCheck=const1, constEvictionCheck=const2)
    else:
        raise NameError(f"Undefined algorithm name: '{config['algorithm']}'")

    metric = main(algorithm=algorithm, environment=env, T=T_horizon, 
                  num_repetitions=config['opt_num_repetitions'], config=config)
    cum_regret= float(np.mean([d["R"][-1] for d in metric.values()]))
    return cum_regret
    

if __name__ == "__main__":
    """ read in data of problem instance """

    with open('input/configSimulation.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    T_horizon = config['T']
    verbose = config['verbose']
    metrics = []
    opt_params = config['opt_params']

    if not opt_params:
        environment, U, lipschitzFactor = get_env_from_config(config, config['seed'])

        algo = NSIC(K=environment.K, T=T_horizon, L=environment.L, U=U, model=environment.model,
                        lipschitzConst=lipschitzFactor, deltaProb=config['delta_prob'], verbose=verbose,
                        constChangeCheck=config['const_change_NSIC'], constEvictionCheck=config['const_eviction_NSIC'])
        metrics.append(main(algorithm=algo, environment=environment, T=T_horizon, 
                            num_repetitions=config['num_repetitions'], config=config))
    else:
        environment, U, lipschitzFactor = get_env_from_config(config, config['seed'])
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, environment), n_trials=config['opt_num_trials'])

        df = study.trials_dataframe(attrs=("number", "value", "params"))
        df_sorted = df.sort_values("value", ascending=True)
        print(df_sorted)
        print("Best params:", study.best_params)
        print("Best score:", study.best_value)
