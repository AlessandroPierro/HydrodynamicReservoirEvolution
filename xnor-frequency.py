import itertools
import os
from datetime import datetime
from multiprocessing import Manager
from typing import Any
from argparse import ArgumentParser

import numpy as np
from nptyping import NDArray, Float

from KDVEquation import KDVEquation
from MAPElites import MAPElites


def parser():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--n_workers', type=int, default=12,
                        help="Number of processes for parallelization.")
    return parser.parse_args()


def fitness_fn(array: NDArray[Any, Float]) -> float:
    global fitness_fn_cache
    global fitness_function_lock
    x = tuple(array)
    x_hash = hash(x)

    if x_hash not in fitness_fn_cache.keys():

        a, b, c, d, e, f, t0, t1, t2, t3 = x
        eps_cnoidals = [abs(a / 2 - 1e-10), abs(b / 2 - 1e-10)]
        detection_times = [40 + 20 * t for t in [t0, t1, t2, t3]]

        eps_soliton = 1.0
        k_soliton = 0.5
        nu = 0.333
        rho2 = 1.0
        delta = 20
        t_delay = 12.75

        results = []
        for k_cnoidals in itertools.product([c, d], [e, f]):
            kdv = KDVEquation(eps_cnoidals, k_cnoidals,
                              eps_soliton, k_soliton,
                              nu, rho2, delta, t_delay,
                              detection_times)
            results.append(kdv.solve())
        with fitness_function_lock:
            fitness_fn_cache.update(
                {x_hash: (x, np.abs(np.linalg.det(np.array(results))))})

    return fitness_fn_cache[x_hash][1]


if __name__ == '__main__':

    args = parser()

    fitness_fn_manager = Manager()
    fitness_fn_cache = fitness_fn_manager.dict()
    fitness_function_lock = fitness_fn_manager.Lock()

    def initializer(rng: np.random.Generator) -> NDArray[Any, Float]:
        return rng.uniform(lb, ub)

    def feature_fn(x: NDArray[Any, Float]) -> NDArray[Any, Float]:
        a, b, c, d, e, f, t0, t1, t2, t3 = tuple(x)
        return np.array([float(np.mean([c, d, e, f])), float(np.std([c, d, e, f]))])

    feature_lb = np.array([0., 0.])
    feature_ub = np.array([1.0, 1.0])
    feature_grid_size = np.array([50, 50])

    lb = np.array([0.] * 10)
    ub = np.array([1.] * 10)

    def mutation(
            x: NDArray[Any, Float],
            rng: np.random.Generator) -> NDArray[Any, Float]:
        return np.clip(x + rng.normal(0, 0.1, x.shape[0]), lb, ub)

    algorithm = MAPElites(
        initializer=initializer,
        fitness_fn=fitness_fn,
        feature_fn=feature_fn,
        lower_bound=feature_lb,
        upper_bound=feature_ub,
        grid_size=feature_grid_size,
        mutation=mutation,
        n_jobs=args.n_workers)

    algorithm.run(g_steps=2000, e_steps=8000, seed=args.seed)

    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    time = time + "-xnor-frequency"
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{time}', exist_ok=True)

    with open(f'results/{time}/archive.csv', 'w') as file:
        file.write("a,b,c,d,e,f,t0,t1,t2,t3,x1,x2,fitness\n")
        for key, value in algorithm.archive.items():
            a, b, c, d, e, f, t0, t1, t2, t3 = value[0]
            x1, x2 = key
            file.write(
                f"{a},{b},{c},{d},{e},{f},{t0},{t1},{t2},{t3},{x1},{x2},{value[1]}\n")

    with open(f'results/{time}/individuals.csv', 'w') as file:
        file.write("a,b,c,d,e,f,t0,t1,t2,t3,fitness\n")
        for key, value in fitness_fn_cache.items():
            a, b, c, d, e, f, t0, t1, t2, t3 = value[0]
            file.write(
                f"{a},{b},{c},{d},{e},{f},{t0},{t1},{t2},{t3},{value[1]}\n")
