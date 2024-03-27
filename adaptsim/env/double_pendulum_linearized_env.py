import numpy as np
import scipy
import scipy.integrate as integrate


class DoublePendulumLinearizedEnv():

    def __init__(
        self,
        render=False,
        dt=0.005,
        #
        max_t=2,
        num_step_eval=20,
    ):
        self.max_t = max_t
        if dt > 0:
            self.max_step = int(max_t / dt) - 1
        else:
            self.num_step_eval = num_step_eval
        self.dt = dt
        self.flag_setup = False

    def reset(self, task=None):
        """
        Set up simulator if first time. Reset task.
        """
        # Save task
        if task is not None:
            self.task = task
        self.num_step_eval = task.num_step_eval

        # Get true system
        m1, m2 = task.true_m
        b1, b2 = task.true_b
        m1 = min(max(0.5, m1), 10)
        m2 = min(max(0.5, m2), 10)
        b1 = min(max(0, b1), 10)
        b2 = min(max(0, b2), 10)

        A = self.get_A(m1, m2, b1, b2)
        B = self.get_B(m1, m2, b1, b2)
        self.A = A
        self.B = B
        self.x = np.array(task.init_x)
        self.Q = np.diag([task.Q_gain for _ in range(4)])
        self.R = np.diag([1, 1])

        return self._get_obs()

    def get_A(self, m1, m2, b1, b2):
        A = np.zeros((4, 4))
        A[0, 2] = 1  # diagonal at top right
        A[1, 3] = 1
        A[2, 0] = 9.81
        A[2, 1] = -9.81 * m2 / m1
        A[3, 0] = -9.81
        A[3, 1] = 9.81 + 2*9.81*m2/m1
        A[2, 2] = -b1 / m1
        A[2, 3] = 2 * b2 / m1
        A[3, 2] = 2 * b1 / m1
        A[3, 3] = -b2 / m2 - 4*b2/m1
        if self.dt > 0:
            A = A * self.dt + np.eye(4)  # convert to discrete
            A[0, 0] += 9.81 * self.dt**2
            A[0, 1] -= 9.81 * self.dt**2
            A[1, 0] -= 9.81 * self.dt**2
            A[1, 1] += 3 * 9.81 * self.dt**2

            A[0, 2] = self.dt - self.dt**2
            A[0, 3] = 2 * self.dt**2
            A[1, 2] = 2 * self.dt**2
            A[1, 3] = self.dt - 5 * self.dt**2
        return A

    def get_B(self, m1, m2, b1, b2):
        B = np.zeros((4, 2))
        B[2, 0] = 1 / m1
        B[2, 1] = -2 / m1
        B[3, 0] = -2 / m1
        B[3, 1] = 1/m2 + 4/m1
        if self.dt > 0:
            B *= self.dt  # convert to discrete
            B[:2, :] = B[2:, :] * self.dt
        return B

    def get_optimal_K(self):
        if self.dt > 0:
            P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
            return np.linalg.multi_dot([
                np.linalg.
                inv(self.R + np.linalg.multi_dot([self.B.T, P, self.B])),
                self.B.T, P, self.A
            ])
        else:
            P = scipy.linalg.solve_continuous_are(
                self.A, self.B, self.Q, self.R
            )
            return np.linalg.multi_dot([np.linalg.inv(self.R), self.B.T, P])

    @property
    def parameter(self):
        return self.task.true_m + self.task.true_b

    def step(self, action=None):
        if action is None:
            K = self.get_optimal_K()
        else:
            K = action
        if self.dt > 0:
            return self.step_discrete(K)
        else:
            return self.step_continuous(K)

    def step_continuous(self, K):
        x_all = [[np.copy(self.x)]]
        u_all = []

        def system(t, x):
            u = -K @ x
            xdot = self.A.dot(x) + self.B.dot(u)
            return xdot

        ret = integrate.solve_ivp(
            system, [0, self.max_t], self.x,
            t_eval=np.linspace(0, self.max_t, self.num_step_eval)
        )
        x_all = ret.y
        u_all = -K @ x_all
        cost = np.sum(np.diagonal(x_all.T.dot(self.Q).dot(x_all))) + np.sum(
            np.diagonal(u_all.T.dot(self.R).dot(u_all))
        )  # negative sign

        info = {}
        info['x'] = x_all.T
        info['u'] = u_all.T
        info['param'] = self.parameter
        info['reward'] = -cost
        done = True
        return self._get_obs(), -cost, done, info

    def step_discrete(self, K):

        x_all = [[np.copy(self.x)]]
        u_all = []
        cost = 0

        for _ in range(self.max_step):
            u = -K.dot(np.copy(self.x))  # u = -K*x
            self.x = self.A.dot(self.x) + self.B.dot(u)  # discrete
            x_copy = np.copy(self.x)
            u_copy = np.copy(u)
            x_all += [x_copy]
            u_all += [u_copy]
            cost += np.linalg.multi_dot(
                [x_copy.T, self.Q, x_copy]
            ) + np.linalg.multi_dot([u_copy.T, self.R, u_copy])

        # Get gradient here
        x = np.vstack(x_all)
        corr = x.T.dot(x)

        # Solve for P in discrete lyapunov equation - AᵀXA - X + Q = 0
        M1 = (self.A - self.B.dot(K)).T  # be careful with the transpose...
        M2 = self.Q + K.T.dot(self.R).dot(K)
        P = scipy.linalg.solve_discrete_lyapunov(M1, M2)

        # Get gradient
        M3 = self.R + np.linalg.multi_dot([self.B.T, P, self.B])
        M4 = 2 * (M3.dot(K) - np.linalg.multi_dot([self.B.T, P, self.A]))
        grad = M4.dot(corr)

        # Return info
        info = {}
        info['x'] = x
        info['param'] = self.parameter
        info['grad'] = grad
        done = True
        return self._get_obs(), -cost, done, info

    def _get_obs(self):
        return np.ones((2, 2))

    def close(self):
        pass

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        return [seed]
