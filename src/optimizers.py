"""Parameter-update rules.

All optimizers update the parameter arrays *in place* from a dict of gradients.

  SGD (momentum=0)   W <- W - lr * dW                       (plain gradient descent)
  SGD with momentum  v <- mu*v - lr*dW ;  W <- W + v        (classical momentum)
  Adam               adaptive per-parameter step sizes      (for experiments)
"""
import numpy as np


class SGD:
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr, self.momentum = lr, momentum
        self.velocity: dict = {}

    def step(self, params: dict, grads: dict) -> None:
        for k, g in grads.items():
            if self.momentum > 0.0:
                v = self.velocity.get(k)
                if v is None:
                    v = np.zeros_like(params[k])
                v = self.momentum * v - self.lr * g
                self.velocity[k] = v
                params[k] += v
            else:
                params[k] -= self.lr * g


class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k, g in grads.items():
            m = self.m.get(k, np.zeros_like(g))
            v = self.v.get(k, np.zeros_like(g))
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g * g
            self.m[k], self.v[k] = m, v
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def get_optimizer(name: str, lr: float, momentum: float = 0.9):
    name = name.lower()
    if name == "sgd":
        return SGD(lr, momentum=0.0)
    if name in ("momentum", "sgd_momentum"):
        return SGD(lr, momentum=momentum)
    if name == "adam":
        return Adam(lr)
    raise ValueError(f"Unknown optimizer '{name}' (use sgd, momentum or adam)")
