"""Interacting Multiple Model filter with CV and CT (coordinated turn) models.

Shared state ``x = [px, py, vx, vy, omega]`` (world frame, Cartesian
velocity, so there is no angle wrapping in the state):

* CV: straight line; omega is driven to 0.
* CT: constant speed turning at ``omega`` (EKF linearisation).

The same filter is the tracker's state estimator *and* the tier-1 predictor
(``rollout``), so prediction needs no second filter.
"""
import numpy as np

STATE_DIM = 5


def _dwna_block(dt, q_acc):
    """Discrete white-noise acceleration covariance for one (pos, vel) axis."""
    q = q_acc ** 2
    return q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])


def _pv_noise(dt, q_acc):
    Q = np.zeros((STATE_DIM, STATE_DIM))
    b = _dwna_block(dt, q_acc)
    for i in (0, 1):   # x: (0, 2), y: (1, 3)
        idx = [i, i + 2]
        Q[np.ix_(idx, idx)] = b
    return Q


class CVModel:
    name = "cv"

    def __init__(self, q_acc=1.0, q_omega=1e-3):
        self.q_acc, self.q_omega = q_acc, q_omega

    def F(self, x, dt):
        F = np.eye(STATE_DIM)
        F[0, 2] = F[1, 3] = dt
        F[4, 4] = 0.0
        return F

    def predict(self, x, P, dt):
        F = self.F(x, dt)
        Q = _pv_noise(dt, self.q_acc)
        Q[4, 4] = self.q_omega ** 2
        return F @ x, F @ P @ F.T + Q


class CTModel:
    name = "ct"

    def __init__(self, q_acc=1.0, q_omega=0.2):
        self.q_acc, self.q_omega = q_acc, q_omega

    @staticmethod
    def f(x, dt):
        px, py, vx, vy, w = x
        if abs(w) < 1e-4:
            # first-order expansion in w (keeps f consistent with the Jacobian below)
            return np.array([px + vx * dt - vy * w * dt ** 2 / 2, py + vy * dt + vx * w * dt ** 2 / 2,
                             vx - vy * w * dt, vy + vx * w * dt, w])
        s, c = np.sin(w * dt), np.cos(w * dt)
        return np.array([
            px + (s * vx - (1 - c) * vy) / w,
            py + ((1 - c) * vx + s * vy) / w,
            c * vx - s * vy,
            s * vx + c * vy,
            w,
        ])

    @staticmethod
    def F(x, dt):
        _, _, vx, vy, w = x
        F = np.eye(STATE_DIM)
        if abs(w) < 1e-4:
            F[0, 2] = F[1, 3] = dt
            F[0, 4] = -vy * dt ** 2 / 2
            F[1, 4] = vx * dt ** 2 / 2
            F[2, 4] = -vy * dt
            F[3, 4] = vx * dt
            return F
        s, c = np.sin(w * dt), np.cos(w * dt)
        F[0, 2], F[0, 3] = s / w, -(1 - c) / w
        F[1, 2], F[1, 3] = (1 - c) / w, s / w
        F[2, 2], F[2, 3] = c, -s
        F[3, 2], F[3, 3] = s, c
        a = (dt * c * w - s) / w ** 2      # d(s/w)/dw
        b = (dt * s * w - (1 - c)) / w ** 2  # d((1-c)/w)/dw
        F[0, 4] = a * vx - b * vy
        F[1, 4] = b * vx + a * vy
        F[2, 4] = -dt * s * vx - dt * c * vy
        F[3, 4] = dt * c * vx - dt * s * vy
        return F

    def predict(self, x, P, dt):
        F = self.F(x, dt)
        Q = _pv_noise(dt, self.q_acc)
        Q[4, 4] = self.q_omega ** 2 * dt
        return self.f(x, dt), F @ P @ F.T + Q


def build_model(spec):
    kind, *args = spec
    return {"cv": CVModel, "ct": CTModel}[kind](*args)


class IMM:
    def __init__(self, models, stay_prob=0.95, dt_ref=0.1):
        self.models = list(models)
        M = len(self.models)
        if M == 1:
            self.Pi_ref = np.ones((1, 1))
        else:
            self.Pi_ref = np.full((M, M), (1 - stay_prob) / (M - 1))
            np.fill_diagonal(self.Pi_ref, stay_prob)
        self.dt_ref = dt_ref
        self.mu = np.full(M, 1.0 / M)
        self.x = np.zeros((M, STATE_DIM))
        self.P = np.tile(np.eye(STATE_DIM), (M, 1, 1))

    def initialize(self, x0, P0, mu0=None):
        M = len(self.models)
        self.x = np.tile(np.asarray(x0, dtype=np.float64), (M, 1))
        self.P = np.tile(np.asarray(P0, dtype=np.float64), (M, 1, 1))
        self.mu = np.full(M, 1.0 / M) if mu0 is None else np.asarray(mu0, dtype=np.float64)

    def copy(self):
        o = IMM.__new__(IMM)
        o.models, o.Pi_ref, o.dt_ref = self.models, self.Pi_ref, self.dt_ref
        o.mu, o.x, o.P = self.mu.copy(), self.x.copy(), self.P.copy()
        return o

    def transition(self, dt):
        """Mode transition matrix for an arbitrary step (stay prob scales with dt)."""
        M = len(self.models)
        if M == 1:
            return self.Pi_ref
        d = np.diag(self.Pi_ref) ** (max(dt, 1e-3) / self.dt_ref)
        Pi = self.Pi_ref.copy()
        for i in range(M):
            off = Pi[i].copy()
            off[i] = 0.0
            Pi[i] = off / max(off.sum(), 1e-12) * (1 - d[i])
            Pi[i, i] = d[i]
        return Pi

    def predict(self, dt):
        if dt <= 0:
            return
        Pi = self.transition(dt)
        c = Pi.T @ self.mu
        W = (Pi * self.mu[:, None]) / np.maximum(c[None, :], 1e-12)   # W[i, j] = P(i | j)
        x_mix = W.T @ self.x
        P_mix = np.empty_like(self.P)
        for j in range(len(self.models)):
            d = self.x - x_mix[j]
            P_mix[j] = np.einsum("i,ijk->jk", W[:, j], self.P + d[:, :, None] * d[:, None, :])
        for j, m in enumerate(self.models):
            self.x[j], self.P[j] = m.predict(x_mix[j], P_mix[j], dt)
        self.mu = c / c.sum()

    def update(self, z, H, R):
        z = np.asarray(z, dtype=np.float64)
        lik = np.empty(len(self.models))
        I = np.eye(STATE_DIM)
        for j in range(len(self.models)):
            x, P = self.x[j], self.P[j]
            y = z - H @ x
            S = H @ P @ H.T + R
            S_inv = np.linalg.inv(S)
            K = P @ H.T @ S_inv
            self.x[j] = x + K @ y
            A = I - K @ H
            self.P[j] = A @ P @ A.T + K @ R @ K.T
            _, logdet = np.linalg.slogdet(2 * np.pi * S)
            lik[j] = -0.5 * (y @ S_inv @ y + logdet)
        m = lik.max()
        w = self.mu * np.exp(lik - m)
        self.mu = np.maximum(w / w.sum(), 1e-4)
        self.mu /= self.mu.sum()
        return m + np.log(w.sum())

    @property
    def state(self):
        x = self.mu @ self.x
        d = self.x - x
        P = np.einsum("j,jkl->kl", self.mu, self.P + d[:, :, None] * d[:, None, :])
        return x, P

    def rollout(self, horizon, step):
        """Open-loop prediction.

        Returns dict with
          ``mode_means`` (M, T, 5) / ``mode_covs`` (M, T, 5, 5): each model run
              on its own (e.g. "keeps turning" vs "goes straight"),
          ``mode_probs`` (M,): current mode probabilities,
          ``mean`` (T, 5) / ``cov`` (T, 5, 5): full IMM prediction (mixing each step).
        """
        n = int(round(horizon / step))
        M = len(self.models)
        mode_means = np.zeros((M, n, STATE_DIM))
        mode_covs = np.zeros((M, n, STATE_DIM, STATE_DIM))
        for j, m in enumerate(self.models):
            x, P = self.x[j].copy(), self.P[j].copy()
            for k in range(n):
                x, P = m.predict(x, P, step)
                mode_means[j, k], mode_covs[j, k] = x, P
        f = self.copy()
        mean = np.zeros((n, STATE_DIM))
        cov = np.zeros((n, STATE_DIM, STATE_DIM))
        for k in range(n):
            f.predict(step)
            mean[k], cov[k] = f.state
        return dict(mode_means=mode_means, mode_covs=mode_covs, mode_probs=self.mu.copy(),
                    mean=mean, cov=cov, times=step * np.arange(1, n + 1))
