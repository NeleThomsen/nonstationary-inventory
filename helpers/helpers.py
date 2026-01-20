import random
from .distributionClass import *

def get_env_from_config(config, seed) -> Tuple[Environment, float, float]:
    # validate inputs
    T = int(config['T'])
    L = int(config['L'])
    h = config['h']
    b = config['b']
    if T <= 0:
        raise ValueError(f"The value of `T` must be non-positive, got {T}")
    if L < 0:
        raise ValueError(f"The value of `lead_time` must be non-negative, got {L}")
    if h < 0:
        raise ValueError(f"The value of `h` must be non-negative, got {h}")
    if b < 0:
        raise ValueError(f"The value of `b` must be non-negative, got {b}")
    
    model = config['model'].lower()
    allowed_models = {"lost_sales", "backlog"}
    if model not in allowed_models:
        raise ValueError(f"`model` must be one of {allowed_models}, got '{config['model']}'")

    # compute parameters
    critical_ratio = b / (b + h)
    lipschitzFactor = max(h, b)

    # read in demand distributions
    S = config['S']
    d_distribution = config['d_distribution']
    d_dists = []
    if d_distribution == "Normal":
        d_means = [random.uniform(1.0, 100.0) for _ in range(S)]
        d_std = 20.0
        d_dists = [Normal(mean_d, d_std, seed) for mean_d in d_means]
    elif d_distribution == "Poisson":
        d_means = [random.uniform(1.0, 100.0) for _ in range(S)]
        d_dists = [Poisson(mean_d, seed) for mean_d in d_means]
    elif d_distribution == "Uniform":
        low_values = [random.uniform(1.0, 100.0) for _ in range(S)]
        scale_values = [random.uniform(0.0, 50.0) for _ in range(S)]
        d_dists = [Uniform(low, scale, seed) for low, scale in zip(low_values, scale_values)]
    elif d_distribution == "Exponential":
        rate_values = [random.uniform(1.0, 100.0) for _ in range(S)]
        d_dists = [Exponential(1/rate, seed) for rate in rate_values]
    else:
        raise ValueError(f"Unknown distribution: {d_distribution}")
    
    max_opt_bs_level = max([d_dist.get_opt_bs_level(critical_ratio, L) for d_dist in d_dists])
    U = 1.2 * float(max_opt_bs_level)  # upper bound on opt. base stock levels
    gamma = pow(T, config['gamma_exponent'])
    K = ceil(U / gamma) + 1
    bslevels = np.linspace(0.0, U, num=K)  # bslevel[0] = 0.0, ... bslevel[K-1] = U
    environment = Environment(model=model, K=K, gamma=gamma, bslevels=bslevels, h=h, b=b, d_dists=d_dists, L=L)

    return environment, U, lipschitzFactor


def get_best_fixed_arm_hindsight(
    environment: Environment,
    T: int,
    changes: Optional[List[int]]) -> Tuple[int, float]:
    """
    Compute the best fixed arm in hindsight given known change points and
    expected cost vectors for each stationary segment.
    """
    changes = changes or []

    # Compute segment lengths
    segment_lengths: List[int] = []
    start = 0
    for change_point in changes:
        segment_lengths.append(change_point - start)
        start = change_point
    segment_lengths.append(T - start)

    # Aggregate total expected cost per arm
    total_cost_per_arm = np.zeros(environment.K, dtype=float)

    for seg_len, exp_costs in zip(segment_lengths, environment.exp_costs_list):
        total_cost_per_arm += seg_len * np.asarray(exp_costs, dtype=float)

    best_arm: int = int(np.argmin(total_cost_per_arm))
    best_avg_cost: float = float(total_cost_per_arm[best_arm] / T)

    return best_arm, best_avg_cost