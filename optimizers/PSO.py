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

        lb = np.tile(A=np.array(lb).reshape((-1, 1)), reps=(1, num_particles))
        ub = np.tile(A=np.array(ub).reshape((-1, 1)), reps=(1, num_particles))

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
        return func(fcn, num_particles, max_iter, inertia_weight, cognitive_weight, social_weight, V_max, lb, ub,
                    int_idx)

    return wrapper


@input_checker
class PSO:
    def __init__(self, fcn, num_particles, max_iter, inertia_weight,
                 cognitive_weight, social_weight, V_max, lb, ub, int_idx):
        """

        :param fcn:
        :param num_particles:
        :param max_iter:
        :param inertia_weight:
        :param cognitive_weight:
        :param social_weight:
        :param V_max:
        :param lb:
        :param ub:
        :param int_idx:
        """
        # the function used for the evaluation (callable)
        self.fcn = fcn
        # the dimension of the optimization problem
        self.dim = len(lb)
        # The indexes of integers
        self.int_idx = int_idx
        # building mask indexing for continuous and discrete variables
        self.continuous_mask = np.ones((self.dim,), dtype=bool)
        self.continuous_mask[self.int_idx] = False
        self.discrete_mask = ~self.continuous_mask
        # initializing the number of particles and iterations
        self.num_particles = num_particles
        self.max_iter = max_iter
        # initializing searching weights & maximum velocity of movement
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.V_max = V_max
        # initializing the lower and upper bound of parameters (dim x particles)
        self.lb = lb
        self.ub = ub

        # initialization of the best metric value of each particle & parameters
        self.metric_best_locals = np.ones((self.num_particles,)) * np.inf  # (particles)
        self.param_best_locals = np.ones((self.dim, self.num_particles)) * np.inf  # (dim x particles)

        # initializing the track of metrics & positions & velocities
        self.metric_track = np.ones((self.num_particles, self.max_iter)) * np.inf
        self.positions = np.zeros((self.dim, self.num_particles, 1))
        self.positions[:, :, 0:1] = (np.random.uniform(low=0, high=1, size=(self.dim, self.num_particles)) * (
                self.ub - self.lb) + self.lb)[:, :, np.newaxis]
        self.positions[self.discrete_mask, :, 0:1] = np.round(self.positions[self.discrete_mask, :, 0:1])

        self.velocities = np.zeros((self.dim, self.num_particles, self.max_iter))
        self.positions = np.minimum(np.maximum(self.positions[:, :, 0], self.lb), self.ub)[:, :, np.newaxis]

        # the current values of metrics of all particles
        self.metric_current = np.ones((1, self.num_particles)) * np.inf
        # best obtained metric
        self.metric_best_global = np.inf

        # initializing best values of parameters (positions)
        self.global_best_position = np.tile(A=self.positions[:, 0, 0].reshape((-1, 1)), reps=(1, self.num_particles))
        self.local_best_position = self.positions[:, :, 0]

    def bounding_positions(self):
        ...
        return

    def get_best_values(self, i):
        best_local_index = self.metric_current < self.metric_best_locals
        self.metric_best_locals = self.metric_current[best_local_index]
        self.local_best_position = self.positions[:, best_local_index[0, :], i]

        min_index = self.metric_current.argmin(axis=1)
        if self.metric_current[0, min_index] < self.metric_best_global:
            self.metric_best_global = self.metric_current[0, min_index]
            self.global_best_position = self.positions[:, min_index, i]


        return

    def run(self):
        for i in range(self.max_iter-1):
            self.metric_current = self.fcn(self.positions[:, :, i])
            self.get_best_values(i)
            self.velocities[:, :, i+1:i+2] = self.inertia_weight * self.velocities[:, :, i:i+1] +

if __name__ == "__main__":
    d = PSO(fcn=f, max_iter=1000, num_particles=45, inertia_weight=0.5, cognitive_weight=1, social_weight=2.0, V_max=10)
    d.run()
