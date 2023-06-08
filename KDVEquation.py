import math
from typing import Any, List

import numpy as np
from nptyping import NDArray, Float
from scipy.fftpack import diff as psdiff
from scipy.integrate import odeint


class KDVEquation:
    """Instantiate and solve a Korteweg–De Vries equation on a 2D domain."""

    @staticmethod
    def _check_preconditions(eps_cnoidals: List[float],
                             k_cnoidals: List[float],
                             eps_soliton: float,
                             k_soliton: float,
                             nu: float,
                             rho2: float,
                             delta: float,
                             t_delay: float,
                             ts_detection: List[float]) -> None:
        assert all(x >= 0 for x in eps_cnoidals)
        assert all(x >= 0 for x in k_cnoidals)
        assert eps_soliton >= 0.0
        assert k_soliton >= 0.0
        assert nu >= 0.0
        assert rho2 >= 0.0
        assert delta >= 0.0
        assert t_delay >= 0.0
        assert all(x >= 0.0 for x in ts_detection)
        assert eps_soliton > 2 * max(eps_cnoidals)
        assert abs(eps_soliton - 12 * nu * (k_soliton ** 2)) < 0.001

    @staticmethod
    def _kdv_soliton(x: NDArray[Any, Float], rho: float, phi: float, k: float) -> NDArray[Any, Float]:
        """Profile of the exact solution to the KdV for a single soliton on the real line.

        :param x: position
        :param rho: trough water height at rest
        :param phi: soliton amplitude
        :param k: soliton wavenumber
        """
        return rho + phi * np.cosh(k * x) ** (-2)

    @staticmethod
    def _kdv_cnoidal(x: NDArray[Any, Float], eps: float, k: float, delta: float) -> NDArray[Any, Float]:
        """Profile of the exact solution to the KdV for a single sinusoidal wave on the real line.

        :param x: position
        :param eps: cnoidal wave amplitude
        :param k: cnoidal wave wavenumber
        :param delta: length of cnoidal wave
        """
        return eps * (np.cos(k * x) ** 2) * np.exp(-1 * (2 * x / delta) ** 8)

    @staticmethod
    def _kdv(u: NDArray[Any, Float], t: NDArray[Any, Float], l: float, nu: float) -> NDArray[Any, Float]:
        """Differential equations for the KdV equation, discretized in x.

        :param u: initial conditions
        :param t: time
        :param l: length of the domain
        :param nu: KdV dispersion coefficient
        """
        ux = psdiff(u, period=l)
        uxxx = psdiff(u, period=l, order=3)
        dudt = -1 * u * ux - 1 * nu * uxxx
        return dudt

    @staticmethod
    def _kdv_solution(u0, t, l: float, nu: float):
        """Use odeint to solve the KdV equation on a periodic domain.

        :param u0: initial conditions
        :param t: time
        :param l: length of domain
        :param nu: KdV dispersion coefficient
        """
        return odeint(KDVEquation._kdv, u0, t, args=(l, nu), mxstep=5000)

    def __init__(self, eps_cnoidals: List[float],
                 k_cnoidals: List[float],
                 eps_soliton: float,
                 k_soliton: float,
                 nu: float,
                 rho2: float,
                 delta: float,
                 t_delay: float,
                 ts_detection: List[float]) -> None:
        """Instantiate a KdV equation."""
        KDVEquation._check_preconditions(eps_cnoidals, k_cnoidals, eps_soliton, k_soliton, nu, rho2, delta, t_delay,
                                         ts_detection)

        self.eps_cnoidals = eps_cnoidals
        self.k_cnoidals = k_cnoidals
        self.eps_soliton = eps_soliton
        self.k_soliton = k_soliton
        self.nu = nu
        self.rho2 = rho2
        self.delta = delta
        self.t_delay = t_delay
        self.ts_detection = ts_detection

        self.ts_detection_discrete = [int(math.floor(float(x) * 10)) for x in ts_detection]
        self.v_soliton = (3 * rho2 + 2 * eps_soliton - 12 * nu * (k_soliton ** 2)) / 3
        self.delay = self.t_delay * self.v_soliton
        self.L = 100.0 * 4
        self.N = 256 * 4

    def solve(self) -> NDArray[Any, Float]:
        """Solve the instantiated KdV equation, returning the state value at the chosen time instants."""
        X = np.linspace(-0.5 * self.L, 0.5 * self.L, self.N + 1)
        t = np.linspace(0, 100, 1001)

        u0 = KDVEquation._kdv_soliton(X + self.delay, self.rho2, self.eps_soliton, self.k_soliton)

        for eps, k in zip(self.eps_cnoidals, self.k_cnoidals):
            u0 += KDVEquation._kdv_cnoidal(X + self.delay, eps, k, self.delta)

        full_solution = KDVEquation._kdv_solution(u0, t, self.L, self.nu)
        solution = full_solution[:, int(self.N / 2):int(self.N / 2) + 256]
        uc = solution[:, 128]

        return list(map(lambda i: uc[i], self.ts_detection_discrete))


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    # Set equation parameters
    eps_cnoidals = [0, 0]
    k_cnoidals = [0.433, 0.5]
    eps_soliton = 1.0
    k_soliton = 0.5
    nu = 0.333
    rho2 = 1.0
    delta = 20
    t_delay = 12.75
    ts_detection = list(np.linspace(35, 70, 101))

    # Instantiate equation
    kdv = KDVEquation(eps_cnoidals=eps_cnoidals,
                      k_cnoidals=k_cnoidals,
                      eps_soliton=eps_soliton,
                      k_soliton=k_soliton,
                      nu=nu,
                      rho2=rho2,
                      delta=delta,
                      t_delay=t_delay,
                      ts_detection=ts_detection)

    # Solve equation
    solution = kdv.solve()

    # Plot results
    plt.plot(ts_detection, solution)
    plt.show()
