import os
from datetime import datetime
from multiprocessing import Manager
from typing import Any
from argparse import ArgumentParser

import numpy as np
from nptyping import NDArray, Float

from KDVEquation import KDVEquation
from MAPElites import MAPElites

N_POINTS = 8


def parser():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--n_workers', type=int, default=12,
                        help="Number of processes for parallelization.")
    return parser.parse_args()


def fitness_fn(array: NDArray[Any, Float]) -> float:
    x = tuple(array)
    x_hash = hash(x)

    if x_hash not in FITNESS_FN_CACHE.keys():
        xs = list(np.round(np.linspace(0, 1, N_POINTS, endpoint=True), 5))
        eps_coeffs = [x * (0.49999 / 9) for x in array[0:6]]
        k_cnoidals = array[6:12]

        results = []
        for xx in xs:
            xs_digits = [int(digit) for digit in str(xx).replace('.', '')]
            xs_digits = xs_digits + [0] * (6 - len(xs_digits))
            eps_cnoidals = [coeff * digit for coeff,
                            digit in zip(eps_coeffs, xs_digits)]

            kdv = KDVEquation(eps_cnoidals=eps_cnoidals,
                              k_cnoidals=k_cnoidals,
                              eps_soliton=1.0,
                              k_soliton=0.5,
                              nu=0.333,
                              rho2=1.0,
                              delta=20,
                              t_delay=12.75,
                              ts_detection=[40 + 20 * t for t in array[20:]])

            results.append(kdv.solve())

        results = np.array(results).T

        with FITNESS_FN_LOCK:
            FITNESS_FN_CACHE.update(
                {x_hash: (x, np.abs(np.linalg.det(np.array(results))))})

    return FITNESS_FN_CACHE[x_hash][1]


if __name__ == '__main__':

    args = parser()

    FITNESS_FN_MANAGER = Manager()
    FITNESS_FN_CACHE = FITNESS_FN_MANAGER.dict()
    FITNESS_FN_LOCK = FITNESS_FN_MANAGER.Lock()

    def initializer(rng: np.random.Generator) -> NDArray[Any, Float]:
        return rng.uniform(lb, ub)

    def feature_fn(array: NDArray[Any, Float]) -> NDArray[Any, Float]:
        xs = list(np.round(np.linspace(0, 1, N_POINTS, endpoint=True), 5))
        eps_coeffs = [x * (0.49999 / 9) for x in array[0:6]]
        eps_cnoidals = []
        for xx in xs:
            xs_digits = [int(digit) for digit in str(xx).replace('.', '')]
            xs_digits = xs_digits + [0] * (6 - len(xs_digits))
            eps_cnoidals.append(
                [coeff * digit for coeff, digit in zip(eps_coeffs, xs_digits)])
        eps_cnoidals_mean = np.mean(np.array(eps_cnoidals).flatten())
        eps_cnoidals_std = np.std(np.array(eps_cnoidals).flatten())
        return np.array([eps_cnoidals_mean, eps_cnoidals_std])

    feature_lb = np.array([0., 0.])
    feature_ub = np.array([0.5, 0.5])
    feature_grid_size = np.array([100, 100])

    lb = np.array([0.] * 28)
    ub = np.array([1.] * 28)

    def mutation(x: NDArray[Any, Float], rng: np.random.Generator) -> NDArray[
            Any, Float]:
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

    algorithm.run(g_steps=2000, e_steps=3000, seed=args.seed)

    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    time = time + "-regression"
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{time}', exist_ok=True)

    with open(f'results/{time}/archive.csv', 'w') as file:
        file.write(
            "eps0,eps1,eps2,eps3,eps4,eps5,k0,k1,k2,k3,k4,k5,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,x1,x2,fitness\n")
        for key, value in algorithm.archive.items():
            t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15 = value[
                0][12:]
            array = value[0]
            eps0, eps1, eps2, eps3, eps4, eps5 = [x * (0.49999 / 9) for x in array[0:6]]
            k0, k1, k2, k3, k4, k5 = array[6:12]
            x1, x2 = key
            file.write(
                f"{eps0},{eps1},{eps2},{eps3},{eps4},{eps5},{k0},{k1},{k2},{k3},{k4},{k5},{t0},{t1},{t2},{t3},{t4},{t5},{t6},{t7},{t8},{t9},{t10},{t11},{t12},{t13},{t14},{t15},{x1},{x2},{value[1]}\n")

    with open(f'results/{time}/individuals.csv', 'w') as file:
        file.write(
            "eos0,eos1,eos2,eos3,eos4,eos5,k0,k1,k2,k3,k4,k5,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,hash,fitness\n")
        for key, value in FITNESS_FN_CACHE.items():
            t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15 = value[
                0][12:]
            array = value[0]
            eps0, eps1, eps2, eps3, eps4, eps5 = [x * (0.49999 / 9) for x in array[0:6]]
            k0, k1, k2, k3, k4, k5 = array[6:12]
            file.write(
                f"{eps0},{eps1},{eps2},{eps3},{eps4},{eps5},{k0},{k1},{k2},{k3},{k4},{k5},{t0},{t1},{t2},{t3},{t4},{t5},{t6},{t7},{t8},{t9},{t10},{t11},{t12},{t13},{t14},{t15},{hash(tuple(value[0]))},{value[1]}\n")
