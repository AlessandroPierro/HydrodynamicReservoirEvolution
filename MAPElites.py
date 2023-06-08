import random
from functools import partial
from multiprocessing import Manager, Pool
from typing import Callable, Any

import numpy as np
from nptyping import NDArray, Float, Int
from numpy.random import SeedSequence, default_rng


class MAPElites:
    """
    Class implementing the MAPElites illumination algorithm.
    """

    def __init__(
            self,
            initializer: Callable[
                [np.random.Generator], NDArray[Any, Float]],
            fitness_fn: Callable[[NDArray[Any, Float]], float],
            feature_fn: Callable[
                [NDArray[Any, Float]], NDArray[Any, Float]],
            lower_bound: NDArray[Any, Float],
            upper_bound: NDArray[Any, Float],
            grid_size: NDArray[Any, Int],
            mutation: Callable[
                [NDArray[Any, Float], np.random.Generator], NDArray[
                    Any, Float]],
            n_jobs: int = 1) -> None:
        """

        :param initializer: function returning a new random individual. Must accept numpy random Generator through the\
                            keyword argument ``rng``.
        :param fitness_fn: fitness function to be maximized
        :param feature_fn: function extracting the features to be  mapped
        :param lower_bound: lower bound for the feature space
        :param upper_bound: upper bound for the feature space
        :param grid_size: size of the discretized grid over the feature space
        :param mutation: mutation operator. Must accept numpy random Generator through the keyword argument ``rng``.
        :param n_jobs: number of parallel jobs to use
        """

        self.initializer = initializer
        self.fitness_fn = fitness_fn
        self.feature_fn = feature_fn
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.grid_size = grid_size
        self.mutation = mutation
        self.n_jobs = n_jobs

        self.manager = Manager()
        self.archive = self.manager.dict()
        self.counter = self.manager.Value('i', 0)
        self.sum_fitness = self.manager.Value('sum_fitness', 0)
        self.lock = Manager().Lock()
        self.pool = Pool(self.n_jobs)

    def _compute_key(self, x: NDArray[Any, Float]) -> NDArray[Any, Int]:
        features = self.feature_fn(x)
        extension = self.upper_bound - self.lower_bound
        return tuple(np.floor(
            (features - self.lower_bound) / extension * self.grid_size).astype(
            int))

    @staticmethod
    def _parallel_task(
            g_steps: int,
            e_steps: int,
            initializer: Callable[
                [np.random.Generator], NDArray[Any, Float]],
            fitness_fn: Callable[[NDArray[Any, Float]], float],
            feature_fn: Callable[
                [NDArray[Any, Float]], NDArray[Any, Float]],
            mutation: Callable[
                [NDArray[Any, Float], np.random.Generator], NDArray[
                    Any, Float]],
            lower_bound: NDArray[Any, Float],
            upper_bound: NDArray[Any, Float],
            grid_size: NDArray[Any, Int],
            archive: Manager().dict,
            counter: Manager().Value,
            lock: Manager().Lock,
            seed) -> None:
        rng = default_rng(seed)
        initializer = partial(initializer, rng=rng)
        mutation = partial(mutation, rng=rng)

        while True:
            with lock:
                counter_local = int(float(counter.value))
                counter.value += 1

            if counter_local < g_steps:
                individual = initializer()
            elif g_steps <= counter_local < g_steps + e_steps:
                parent, _ = archive[random.choice(list(archive.keys()))]
                individual = mutation(parent)
            else:
                return

            fitness = fitness_fn(individual)
            features = feature_fn(individual)
            key = tuple(
                map(lambda x: int(x), np.floor((features - lower_bound) / (
                    upper_bound - lower_bound) * grid_size)))

            with lock:
                previous_fitness = archive.get(key, (None, float("-inf")))[1]
                if fitness > previous_fitness:
                    archive.update({key: (individual, fitness)})

    def run(self, g_steps: int, e_steps: int, seed: int = 42) -> None:
        """
        Launch the execution of MAPElites.

        :param g_steps: number of exploration steps to run
        :param e_steps: number of mutation steps to run
        :param seed: random seed
        """

        with self.lock:
            self.counter.value = 0

        arguments = (g_steps, e_steps, self.initializer, self.fitness_fn,
                     self.feature_fn, self.mutation, self.lower_bound,
                     self.upper_bound, self.grid_size,
                     self.archive, self.counter, self.lock)

        ss = SeedSequence(seed)
        seeds = ss.spawn(self.n_jobs)
        self.pool.starmap(__class__._parallel_task,
                          [arguments + (seed,) for seed in seeds])

    def update_grid(self,
                    lower_bound: NDArray[Any, Float],
                    upper_bound: NDArray[Any, Float],
                    grid_size: NDArray[Any, Int]) -> None:
        """
        Update the grid over the feature space, without cleaning the archive.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.grid_size = grid_size
        elements = list(self.archive.values())
        with self.lock:
            self.archive.clear()
        keys = list(map(lambda x: self._compute_key(x[0]), elements))
        with self.lock:
            self.archive.update(dict(zip(keys, elements)))

    def __del__(self):
        self.pool.close()
        self.pool.join()
        self.manager.shutdown()
