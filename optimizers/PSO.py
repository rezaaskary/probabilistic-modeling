# %%
import numpy as np

# PSO parameters
num_particles = 20
num_dimensions = 5
max_iter = 100
w = 0.5  # inertia weight
c1 = 1.0  # cognitive weight
c2 = 2.0  # social weight
v_max = 10  # maximum velocity
lb = [0, 1, 1, 1, 1, 2]
ub = [0, 1, 1, 1, 10, 7]
int_idx = [4, 5]


def f(x):
    return ((x[:4, :] ** 2).sum(axis=0) - x[4, :] - x[5, :])[np.newaxis, :]


def input_checker(func):
    def wrapper(fcn: callable = None,
                num_particles: int = 200,
                max_iter: int = 100,
                inertia_weight: float = 0.5,
                cognitive_weight: float = 1.0,
                social_weight: float = 2.0,
                V_max: float = 10,
                lb: list = [0, 1, 1, 1, 1, 2],
                ub: list = [0, 1, 1, 1, 10, 7],
                int_idx: list = [4, 5]):
        if not isinstance(num_particles, int):
            Exception(f'The value of {num_particles} is incorrect')

        if not isinstance(max_iter, int):
            Exception(f'The value of {max_iter} is incorrect')

        if not isinstance(inertia_weight, float):
            Exception(f'The value of {inertia_weight} is incorrect')

        if not isinstance(cognitive_weight, float):
            Exception(f'The value of {cognitive_weight} is incorrect')

        if not isinstance(V_max, float):
            Exception(f'The value of {V_max} is incorrect')
        return func(fcn, num_particles, max_iter, inertia_weight, cognitive_weight, social_weight, V_max, lb, ub)

    return wrapper


@input_checker
class PSO:
    def __init__(self, fcn, num_particles, max_iter, inertia_weight,
                 cognitive_weight, social_weight, V_max, lb, ub, int_idx):

        self.fcn = fcn
        self.dim = len(lb)
        self.int_idx = int_idx
        self.continuous_mask = np.ones((self.dim,), dtype=bool)
        self.continuous_mask[self.int_idx] = False
        self.discrete_mask = ~self.continuous_mask
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.V_max = V_max
        self.lb = np.tile(A=np.array(lb).reshape((-1, 1)), reps=(1, self.num_particles))
        self.ub = np.tile(A=np.array(ub).reshape((-1, 1)), reps=(1, self.num_particles))
        self.metric_best_locals = np.ones((1, self.num_particles)) * np.inf
        self.metric_track = np.ones((self.num_particles, self.max_iter)) * np.inf
        self.metric_current = np.ones((1, self.num_particles)) * np.inf
        self.gbest_value = np.inf
        self.positions = np.zeros((self.dim, self.num_particles, self.max_iter))
        self.velocities = np.zeros((self.dim, self.num_particles, num_dimensions))
        self.positions[:, :, 0:1] = (np.random.uniform(low=0, high=1, size=(self.dim, self.num_particles)) * (
                self.ub - self.lb) + self.lb)[:, :, np.newaxis]

        self.positions[self.discrete_mask, :, 0:1] = np.round(self.positions[self.discrete_mask, :, 0:1])
        self.positions = np.minimum(np.maximum(self.positions[:, :, 0], self.lb), self.ub)[:, :, np.newaxis]
        self.global_best = np.tile(A=self.positions[:, 0, 0].reshape((-1, 1, 1)), reps=(1, self.num_particles, 1))
        self.local_best = self.positions[:, :, 0:1]

    def get_best_values(self):
        self.metric_best_locals = self.metric_current[:, self.metric_current < self.metric_best_locals]
        min_index = self.metric_current.argmin(axis=1)
        if self.metric_current[0, min_index] < self.gbest_value: self.gbest_value = self.metric_current[0, min_index]

        return

    def run(self):
        for i in range(self.max_iter):
            self.metric_current = self.fcn(self.positions[:, :, i])


if __name__ == "__main__":
    d = PSO(fcn=f, max_iter=1000,num_particles=45, inertia_weight=0.5, cognitive_weight=1, social_weight=2.0, V_max=10)
    d.run()
