import numpy as np
# from pprint import pprint as pp
# import copy
# from scipy.optimize import minimize, Bounds
import logging
import logging.config
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import sys
from tqdm import trange

# logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs.log')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


class AllocationAlgGreedy:
    def __init__(
            self,
            num_agents: int,
            rng_seed: int | None = None,
            use_prop_1: bool = False,
            N: int = 50,
            ) -> None:
        
        self.num_agents = num_agents
        self.bundles = None
        self.good_values_list = None
        self.rng = np.random.default_rng(seed = rng_seed)
        self.use_prop_1 = use_prop_1
        self.alpha_tracker = []

        

        if self.num_agents == 2:
            # Set up solution-space
            self.N = N
            self.x = np.linspace(0, 1, self.N)  # array of x-values from 0 to 1
            self.y = np.linspace(0, 1, self.N)  # array of y-values from 0 to 1
            # Create a 2D mesh grid, X and Y will each be shape (N, N).
            self.X, self.Y = np.meshgrid(self.x, self.y)
            self.Z = np.empty((self.N, self.N), dtype=float)

            # Initialize plots.
            self.fig, self.ax = plt.subplots(2, 1, figsize=(6,5), layout='constrained', gridspec_kw={'height_ratios':[2,1]})
            self.colormesh = self.ax[0].pcolormesh(self.X, self.Y, self.Z, cmap='plasma', shading='auto')
            self.ax[0].set_title('t = 0')
            self.ax[0].set_xlabel('x')
            self.ax[0].set_ylabel('y')
            self.cbar = self.fig.colorbar(self.colormesh, ax=self.ax[0])

            self.marker, = self.ax[0].plot(0, 0, marker='x', color='red', markersize=10, mew=2)
            self.line, = self.ax[1].plot([0], [0], 'k-+')

            self.alpha_txt = self.ax[1].text(1, 0.98, rf'$\alpha_\mathrm{{min}}={self.num_agents}$',
                                horizontalalignment='right',
                                verticalalignment='top',
                                transform=self.ax[1].transAxes)
            
            self.ax[1].set_xlabel('$t$')
            self.ax[1].set_ylabel('$\\alpha$')

           
            self.t_list = []


    def allocate_good(
            self,
            new_good_values: list
            ) -> None:
        '''
        Allocate the new good.
        '''
        if self.good_values_list is None:
            self.add_to_bundle(self.rng.choice(range(self.num_agents)), new_good_values)
        else:
            min_value = np.inf
            for agent_index, good_value in enumerate(new_good_values):
                if (self.value_all_goods(agent_index) + good_value) == 0:
                    value = 1
                else:
                    if self.use_prop_1:
                        value = (self.value_bundle(agent_index) + self.value_best_unalloc_good(agent_index)) / (self.value_all_goods(agent_index) + good_value)
                    else:
                        value = self.value_bundle(agent_index) / (self.value_all_goods(agent_index) + good_value)
                if value < min_value:
                    min_agent = [agent_index]
                    min_value = value
                elif value == min_value:
                    min_agent.append(agent_index)

            if len(min_agent) == 1:
                self.add_to_bundle(min_agent, new_good_values)
            elif len(min_agent) > 1:
                self.add_to_bundle(self.rng.choice(min_agent), new_good_values)


    def add_to_bundle(self, agent_index: int, new_good_values: list,) -> None:
        '''
        Add good to bundle of agent specified by agent_index. Each bundle is a boolean list where True means good is allocated to agent.
        '''
        foo = np.zeros([1, self.num_agents], dtype=bool)
        foo[0, agent_index] = True

        if self.good_values_list is None:
            self.bundles = foo.T
            self.good_values_list = np.array([new_good_values])
        else:
            self.bundles = np.append(self.bundles, foo.T, axis=1)
            self.good_values_list = np.append(self.good_values_list, [new_good_values], axis=0)


    def value_bundle(self, agent_index: int) -> float:
        '''
        Returns value of the bundle of goods allocated to the agent.
        '''
        return np.sum(self.good_values_list[:, agent_index][self.bundles[agent_index]])
    

    def value_all_goods(self, agent_index: int) -> float:
        '''
        Returns value of all goods, both allocated and unallocated, for an agent.
        '''
        return np.sum(self.good_values_list[:, agent_index])
    

    def value_best_unalloc_good(self, agent_index: int) -> float:
        '''
        Returns value of the best unallocated good.
        '''
        unallocated_goods = self.good_values_list[:, agent_index][~self.bundles[agent_index]]
        if len(unallocated_goods) == 0:
            return 0.
        else:
            return np.max(unallocated_goods)


    def get_prop1_alpha(self) -> np.ndarray:
        '''
        Returns the ratio: v_i (A_i U {g}) * n / v_i(G).
        '''
        prop1_list = np.zeros(self.num_agents)
        for agent_index in range(self.num_agents):
            if self.value_all_goods(agent_index) == 0:
                prop1_list[agent_index] = self.num_agents
            else:
                prop1_list[agent_index] = (self.value_bundle(agent_index) + self.value_best_unalloc_good(agent_index)) * self.num_agents / self.value_all_goods(agent_index)
        return prop1_list
    

    def test_allocation(self, test_good) -> float:
        '''
        Return alpha for a good, without changing the system.
        '''
        self.allocate_good(test_good)
        foo = self.get_prop1_alpha()[0]
        self.remove_last_good()
        return foo
    

    def find_worst_good(self) -> list:
        '''
        Returns the worst good to add for 2 agents.
        '''
        # worst_good_index = np.argmin(self.Z)
        # if worst_good_index == 0:
        #     return [0.5, 0]
        # else:
        #     return [self.x[worst_good_index % self.N], self.y[worst_good_index // self.N]]

        coords = np.array((self.X.ravel(), self.Y.ravel())).T.reshape(self.N, self.N, self.num_agents)
        worst_good = self.rng.choice(coords[self.Z == np.min(self.Z)].reshape(-1, self.num_agents))
        return [worst_good[0], worst_good[1]]


    def remove_last_good(self) -> None:
        '''
        Remove last good added to the bundles.
        '''
        self.bundles = np.delete(self.bundles, -1, 1)
        self.good_values_list = np.delete(self.good_values_list, -1, 0)


    def update_solution_space(self) -> None:
        '''
        Update all values of alpha for combinations of good values for 2 agents.
        '''
        for i in range(self.N):
            for j in range(self.N):
                self.Z[i, j] = self.test_allocation([self.x[i], self.y[j]])


    def find_next_good(self, t) -> list:
        '''
        Find next good for 2 agents.
        '''
        if t % 10 < 3:
            return [self.rng.uniform(), self.rng.uniform()]
        else:
            return self.find_worst_good()


    def update_fig(self, t: int):
        '''
        Update figure for animation for 2 agents.
        '''
        if not self.num_agents == 2:
            logger.error('update_fig only works for 2 agents.')
        self.update_solution_space()
        next_good = self.find_next_good(t)
        self.allocate_good(next_good)
        self.alpha_tracker.append(self.get_prop1_alpha()[0])
        self.t_list.append(t)

        self.colormesh.set_array(self.Z.ravel())
        self.colormesh.autoscale()
        self.marker.set_data([next_good[0]], [next_good[1]])
        self.ax[0].set_title(f"t = {t}")
        self.line.set_data(self.t_list, self.alpha_tracker)
        self.ax[1].relim()
        self.ax[1].autoscale_view()
        self.alpha_txt.set_text(rf'$\alpha_\mathrm{{min}} = {np.round(np.min(self.alpha_tracker), 2)}$')
        return (self.colormesh, self.marker, self.line, self.alpha_txt)
    

    def init_fig(self):
        '''
        Initialize figure.
        '''
        return (self.colormesh, self.marker, self.line,)


def main():
    alloc = AllocationAlgGreedy(2, rng_seed = 123)

    anim = FuncAnimation(
        alloc.fig,
        alloc.update_fig,
        frames=trange(100, desc='Time step'),
        blit=True,     
        interval=300,
        init_func=alloc.init_fig
    )

    anim.save('allocation_mixer.mp4')

    logger.info(f'Alpha: {alloc.alpha_tracker}')
    logger.info(f'Bundles: {alloc.bundles}')
    logger.info(f'Good values: {alloc.good_values_list}')


if __name__ == '__main__': main()