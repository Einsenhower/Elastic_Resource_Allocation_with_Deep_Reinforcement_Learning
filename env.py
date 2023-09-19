import numpy as np
import gym
from gym import spaces

np.random.seed(2023)

# Define some basic work for the simulation
class Task(object):
    def __init__(self, _id, _resource_num, _weight):
        self.id = _id
        self.resource_num = _resource_num
        self.weight = _weight


class Machine(object):
    def __init__(self, _id, _capacity, p_work):
        self.id = _id
        self.capacity = _capacity
        self.p_work = p_work
        self.p_base = 5


class World(object):
    def __init__(self, task_num, machine_num):
        self.task_num = task_num
        self.machine_num = machine_num
        # Task arrival time
        self.task_reach_time = {0: [i for i in range(15)], 20: [i for i in range(15, 30)],
                                30: [i for i in range(30, 40)]}
        self.task_list = []
        self.machine_list = []
        self.process_time = np.random.random(size=(task_num, machine_num)) * 100

    def data_generator(self):
        resource_range = [5, 10]
        weight_range = [1, 3]
        # weight can be setting or be randomly
        #weight_list = [1, 1, 1, 1, 1, 10, 10, 10, 10, 10]
        for i in range(1, self.task_num + 1):
            task = Task(_id=i,
                        _resource_num=np.random.randint(resource_range[0], resource_range[1] + 1),
                        _weight=np.random.randint(weight_range[0], weight_range[1] + 1))
            self.task_list.append(task)
            # Set here if needed
            # task.weight = weight_list[i - 1]
        capacity_range = [10, 20]
        p_work_range = [10, 20]
        for j in range(1, self.machine_num + 1):
            machine = Machine(_id=j,
                              _capacity=np.random.randint(capacity_range[0], capacity_range[1] + 1),
                              p_work=np.random.randint(p_work_range[0], p_work_range[1] + 1))
            self.machine_list.append(machine)


class Env(gym.Env):
    def __init__(self, world):
        self.task_num = world.task_num
        self.machine_num = world.machine_num
        self.task_list = world.task_list
        self.machine_list = world.machine_list
        self.world = world
        self.new_task_reach_time = list(self.world.task_reach_time.keys())
        self.new_task_reach_idx = 0
        self.state = None
        self.best_obj = None
        # state: task * machine + task(Resource Needed) + Task State + Task Order + task weight + machine remain capacity
        self.state_len = self.task_num * self.machine_num + 4 * self.task_num + self.machine_num
        self.param_dim = self.task_num * self.machine_num
        state_lb = np.ones(self.state_len) * -999
        state_ub = np.ones(self.state_len) * 999
        self.threshold = 0.5
        self.observation_space = spaces.Box(low=state_lb,
                                            high=state_ub,
                                            shape=(self.state_len,),
                                            dtype=float,
                                            seed=2023)
        self.action_space = spaces.Box(low=np.zeros(self.param_dim),
                                       high=np.ones(self.param_dim),
                                       shape=(self.param_dim,),
                                       dtype=float,
                                       seed=2023)

    # Set reset point
    def reset(self):
        self.new_task_reach_idx = 0
        assignment = [0 for j in range(self.machine_num) for i in range(self.task_num)]
        task_resource = [self.task_list[i].resource_num for i in range(self.task_num)]
        task_weight = [self.task_list[i].weight for i in range(self.task_num)]
        task_reach_time = self.world.task_reach_time
        cur_task_num = len(task_reach_time[0])
        task_status = [1 if i < cur_task_num else 0 for i in range(self.task_num)]
        task_order = [0 for _ in range(self.task_num)]
        self.new_task_reach_idx += 1
        machine_info = [self.machine_list[j].capacity for j in range(self.machine_num)]
        self.state = assignment + task_resource + task_status + task_order + task_weight + machine_info
        return self.state

    # Define the first problem the maximum task problem
    def cal_obj1(self, state):
        assigned_task_list = []
        machine_assigned_task_dd = {}
        state_info = np.array(state[0: self.task_num * self.machine_num], dtype=int)
        state_info = state_info.reshape((self.task_num, self.machine_num))
        for i in range(self.task_num):
            if np.sum(state_info[i]) > 0:
                assigned_task_list.append(i)
                machine_idx = np.argmax(state_info[i])
                if machine_idx in machine_assigned_task_dd:
                    machine_assigned_task_dd[machine_idx].append(i)
                else:
                    machine_assigned_task_dd[machine_idx] = [i]
        task_order = np.array(state[self.task_num * self.machine_num + self.task_num + self.task_num:
                                    self.task_num * self.machine_num + self.task_num * 3], dtype=int)
        machine_assigned_task_seq_dd = {}
        for machine in machine_assigned_task_dd.keys():
            task_list = machine_assigned_task_dd[machine]
            task_list_dd = {task_list[i]: task_order[task_list[i]] for i in range(len(task_list))}
            task_list_dd_sorted = sorted(task_list_dd.items(), key=lambda x: x[1])
            machine_assigned_task_seq_dd[machine] = [item[0] for item in task_list_dd_sorted]
        task_info = {}
        for machine in machine_assigned_task_seq_dd.keys():
            task_list = machine_assigned_task_seq_dd[machine]
            i = 0
            cur_t = 0
            task_finish_time_list = []
            task_occupation_capacity = []
            machine_residual_capacity = self.machine_list[machine].capacity
            while i < len(task_list):
                task = task_list[i]
                if self.task_list[task].resource_num <= self.machine_list[machine].capacity:
                    task_process_time = self.world.process_time[task][machine]
                    task_finish_time = cur_t + task_process_time
                    task_finish_time_list.append(task_finish_time)
                    task_occupation_capacity.append(self.task_list[task].resource_num)
                    machine_residual_capacity -= self.task_list[task].resource_num
                    task_info[task] = {"machine": machine, "st": cur_t, "et": task_finish_time}
                    i += 1
                else:
                    cur_t = np.min(task_finish_time_list)
                    idx = np.argmin(task_finish_time_list)
                    machine_residual_capacity += task_occupation_capacity[idx]
                    del task_finish_time_list[idx]
                    del task_occupation_capacity[idx]
        obj = 0
        for task in task_info.keys():
            if task_info[task]["et"] > obj:
                obj = task_info[task]["et"]
        return obj, task_info

    # Define the second problem the minimum power problem
    def cal_obj2(self, state):
        assigned_task_list = []
        machine_assigned_task_dd = {}
        state_info = np.array(state[0: self.task_num * self.machine_num], dtype=int)
        state_info = state_info.reshape((self.task_num, self.machine_num))
        for i in range(self.task_num):
            if np.sum(state_info[i]) > 0:
                assigned_task_list.append(i)
                machine_idx = np.argmax(state_info[i])
                if machine_idx in machine_assigned_task_dd:
                    machine_assigned_task_dd[machine_idx].append(i)
                else:
                    machine_assigned_task_dd[machine_idx] = [i]
        task_order = np.array(state[self.task_num * self.machine_num + self.task_num + self.task_num:
                                    self.task_num * self.machine_num + self.task_num * 3], dtype=int)
        machine_assigned_task_seq_dd = {}
        for machine in machine_assigned_task_dd.keys():
            task_list = machine_assigned_task_dd[machine]
            task_list_dd = {task_list[i]: task_order[task_list[i]] for i in range(len(task_list))}
            task_list_dd_sorted = sorted(task_list_dd.items(), key=lambda x: x[1])
            machine_assigned_task_seq_dd[machine] = [item[0] for item in task_list_dd_sorted]
        task_info = {}
        machine_info = {}
        for machine in machine_assigned_task_seq_dd.keys():
            task_list = machine_assigned_task_seq_dd[machine]
            i = 0
            cur_t = 0
            task_finish_time_list = []
            task_occupation_capacity = []
            machine_residual_capacity = self.machine_list[machine].capacity
            while i < len(task_list):
                task = task_list[i]
                if self.task_list[task].resource_num <= self.machine_list[machine].capacity:
                    task_process_time = self.world.process_time[task][machine]
                    task_finish_time = cur_t + task_process_time
                    task_finish_time_list.append(task_finish_time)
                    task_occupation_capacity.append(self.task_list[task].resource_num)
                    machine_residual_capacity -= self.task_list[task].resource_num
                    task_info[task] = {"machine": machine, "st": cur_t, "et": task_finish_time}
                    i += 1
                else:
                    cur_t = np.min(task_finish_time_list)
                    idx = np.argmin(task_finish_time_list)
                    machine_residual_capacity += task_occupation_capacity[idx]
                    del task_finish_time_list[idx]
                    del task_occupation_capacity[idx]
        max_finish_time = 0
        for task in task_info.keys():
            if task_info[task]["et"] > max_finish_time:
                max_finish_time = task_info[task]["et"]
        obj = sum([self.machine_list[i].p_base * max_finish_time for i in range(self.machine_num)])
        for task in task_info.keys():
            item = task_info[task]
            machine = item['machine']
            st = item['st']
            et = item['et']
            obj += (et - st) * self.machine_list[machine].p_work
        return obj, task_info

    # Define the third problem the VIP clients maximum task problem
    def cal_obj3(self, state):
        assigned_task_list = []
        machine_assigned_task_dd = {}
        state_info = np.array(state[0: self.task_num * self.machine_num], dtype=int)
        state_info = state_info.reshape((self.task_num, self.machine_num))
        for i in range(self.task_num):
            if np.sum(state_info[i]) > 0:
                assigned_task_list.append(i)
                machine_idx = np.argmax(state_info[i])
                if machine_idx in machine_assigned_task_dd:
                    machine_assigned_task_dd[machine_idx].append(i)
                else:
                    machine_assigned_task_dd[machine_idx] = [i]
        machine_info = state[self.task_num * self.machine_num + 2 * self.task_num:]
        if len(np.where(np.array(machine_info) < 0)[0]) > 0:
            machine_punish = len(np.where(np.array(machine_info) < 0)[0])
        else:
            machine_punish = 0
        obj = np.sum([self.task_list[t].weight for t in assigned_task_list])
        task_machine_punish = 0
        for machine in machine_assigned_task_dd.keys():
            for task_idx in machine_assigned_task_dd[machine]:
                if self.task_list[task_idx].resource_num > self.machine_list[machine].capacity:
                    task_machine_punish += 1
        info = {'atl': assigned_task_list,
                'mat': machine_assigned_task_dd,
                'machine_punish': machine_punish,
                'task_machine_punish': task_machine_punish}
        return obj, info

    def step(self, action):
        task_status = self.state[self.task_num * self.machine_num + self.task_num:
                                 self.task_num * self.machine_num + self.task_num + self.task_num]
        if len(np.where(np.array(task_status) == 1)[0]) == 0:
            if self.new_task_reach_idx < len(self.new_task_reach_time):
                # Previously arrived tasks have been fully allocated and need to trigger a reassignment when the next batch of tasks arrives
                reach_time = self.new_task_reach_time[self.new_task_reach_idx]
                new_reach_task_list = self.world.task_reach_time[reach_time]
                self.new_task_reach_idx += 1

                # Calculation of which tasks have been completed and reassignment of uncompleted tasks
                assigned_task_list = []
                machine_assigned_task_dd = {}
                state_info = np.array(self.state[0: self.task_num * self.machine_num], dtype=int)
                state_info = state_info.reshape((self.task_num, self.machine_num))
                for i in range(self.task_num):
                    if np.sum(state_info[i]) > 0:
                        assigned_task_list.append(i)
                        machine_idx = np.argmax(state_info[i])
                        if machine_idx in machine_assigned_task_dd:
                            machine_assigned_task_dd[machine_idx].append(i)
                        else:
                            machine_assigned_task_dd[machine_idx] = [i]
                # Calculating tasks on each machine
                finish_task_list = {}
                for machine in machine_assigned_task_dd.keys():
                    finish_task_list[machine] = []
                    task_list = machine_assigned_task_dd[machine]
                    machine_residual_capacity = self.machine_list[machine].capacity
                    task_idx = []
                    task_finish_time_list = []
                    task_finish_time_list1 = []
                    task_occupation_capacity = []
                    cur_t = 0
                    i = 0
                    while i < len(task_list):
                        if cur_t >= reach_time:
                            break
                        task = task_list[i]
                        assert self.task_list[task].resource_num <= self.machine_list[machine].capacity
                        if self.task_list[task].resource_num <= machine_residual_capacity:
                            task_process_time = self.world.process_time[task][machine]
                            task_finish_time = cur_t + task_process_time
                            task_finish_time_list.append(task_finish_time)
                            task_finish_time_list1.append(task_finish_time)
                            task_occupation_capacity.append(self.task_list[task].resource_num)
                            task_idx.append(task)
                            machine_residual_capacity -= self.task_list[task].resource_num
                            i += 1
                        else:
                            cur_t = np.min(task_finish_time_list1)
                            idx = np.argmin(task_finish_time_list1)
                            machine_residual_capacity += task_occupation_capacity[idx]
                            del task_finish_time_list1[idx]
                            del task_occupation_capacity[idx]

                    for i in range(len(task_idx)):
                        if task_finish_time_list[i] < reach_time:
                            finish_task_list[machine].append(task_idx[i])
                for i in range(self.task_num * self.machine_num):
                    self.state[i] = 0
                for i in range(self.task_num):
                    self.state[self.task_num * self.machine_num + self.task_num + self.task_num + i] = 0
                task_order = 1
                for machine in finish_task_list.keys():
                    for task in finish_task_list[machine]:
                        idx = task * machine
                        self.state[idx] = 1
                        self.state[self.task_num * self.machine_num + self.task_num + idx] = 3
                        self.state[self.task_num * self.machine_num + self.task_num + self.task_num + task] = task_order
                        task_order += 1
                for idx in new_reach_task_list:
                    self.state[self.task_num * self.machine_num + self.task_num + idx] = 1
        else:
            assigned_task_list = np.where(np.array(task_status) >= 2)[0]
            unreached_task_list = np.where(np.array(task_status) == 0)[0]
            # Based on the action output of the
            assignment = action[0: self.task_num * self.machine_num]
            idx = np.argmax(assignment)
            task_idx = int(idx / self.machine_num)
            while task_idx in assigned_task_list or task_idx in unreached_task_list:
                assignment[idx] = -9999
                idx = np.argmax(assignment)
                task_idx = int(idx / self.machine_num)
            # Selected task_idx, calculate machine index
            machine_idx = idx - task_idx * self.machine_num
            assert task_idx < self.task_num and machine_idx < self.machine_num
            self.state[idx] = 1
            self.state[self.task_num * self.machine_num + self.task_num + task_idx] = 2
            task_order = np.max(self.state[self.task_num * self.machine_num + self.task_num + self.task_num:
                                           self.task_num * self.machine_num + self.task_num + self.task_num + self.task_num])
            self.state[self.task_num * self.machine_num + self.task_num + self.task_num + task_idx] = task_order + 1
        if len(np.where(np.array(task_status) >= 2)[0]) == self.task_num:
            # All tasks are assigned and the objective function is computed to get the reward
            done = True
            next_state = self.state.copy()
            # For problme change to 0.0001
            obj, obj_info = self.cal_obj3(next_state)
            reward = 0 - obj * 0.01
            info = {'obj_info': obj_info, 'obj': obj}
        else:
            done = False
            reward = 0
            next_state = self.state.copy()
            info = {}
        return next_state, reward, done, info


if __name__ == "__main__":
    env = Env()
    env.reset()
    print(env.state)
    print(env.observation_space.sample())
    print(env.observation_space.high)
