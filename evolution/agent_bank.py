# -----------------------------------------------------------------------------
#   @brief:
#       The species_bank save all the information of the species
#       written by Tingwu Wang
# -----------------------------------------------------------------------------
import multiprocessing
from agent import evolution_agent
from util import parallel_util


class agent_bank(object):
    '''
        @brief:
            ['policy_weights', 'baseline_weights', 'running_mean_info']
            ['training_stats']
            ['start_time', 'end_time']
            ['env_xml', 'topology_data']
            ['parent_species']
    '''

    def __init__(self, num_agent, args):
        self.args = args
        self.task_queue = multiprocessing.JoinableQueue()
        self.result_queue = multiprocessing.Queue()

        # make the agents
        self.agents_dict = dict()
        for i_agent in range(num_agent):
            agent = evolution_agent.evolutionary_agent(
                args, self.task_queue, self.result_queue, i_agent
            )
            agent.start()
            self.agents_dict[i_agent] = (agent)

    def put(self, task):
        self.task_queue.put(task)

    def join(self):
        self.task_queue.join()

    def get(self):
        results = self.result_queue.get()

        agent_id = results['agent_id']
        del self.agents_dict[agent_id]
        agent = evolution_agent.evolutionary_agent(
            self.args, self.task_queue, self.result_queue, agent_id
        )
        agent.start()
        self.agents_dict[agent_id] = agent
        return results

    def end(self):
        for _ in self.agents_dict:
            self.task_queue.put((parallel_util.END_SIGNAL, None))

        del self.agents_dict

    def log_information(self):
        pass
