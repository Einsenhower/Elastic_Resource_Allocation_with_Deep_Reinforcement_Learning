from env import Env, World
import numpy as np
import pandas as pd

# Set random match algorithm (Baseline)
def random_match():
    # state
    # task * machine + task(Resource needed) + task state + task order + task weight + machine(Resource remain)
    state = env_instance.reset()
    task_order = 1
    for time_key in env_instance.world.task_reach_time.keys():
        task_list = env_instance.world.task_reach_time[time_key]
        for task in task_list:
            machine = np.random.randint(0, env_instance.machine_num)
            idx = task * machine
            state[idx] = 1
            state[env_instance.task_num * env_instance.machine_num + env_instance.task_num + task] = 2
            state[env_instance.task_num * env_instance.machine_num + env_instance.task_num * 2 + task] = task_order
            task_order += 1
    return state


# Set average match algorithm (Baseline)
def avg_match():
    # state
    # task * machine + task(Resource needed) + task state + task order + task weight + machine(Resource remain)
    state = env_instance.reset()
    task_order = 1
    machine_idx = 0
    for time_key in env_instance.world.task_reach_time.keys():
        task_list = env_instance.world.task_reach_time[time_key]
        for task in task_list:
            machine = machine_idx
            idx = task * machine
            state[idx] = 1
            state[env_instance.task_num * env_instance.machine_num + env_instance.task_num + task] = 2
            state[env_instance.task_num * env_instance.machine_num + env_instance.task_num * 2 + task] = task_order
            task_order += 1
            machine_idx += 1
            if machine_idx >= env_instance.machine_num:
                machine_idx = 0
    return state


# Generate the Environment
world = World(task_num=50, machine_num=5)
world.data_generator()

env_instance = Env(world)
state1 = avg_match()
avg_obj = env_instance.cal_obj3(state1)
result = pd.DataFrame()
result['best_obj'] = avg_obj
result['cur_obj'] = avg_obj
result.to_csv("data/avg_task_{}_machine_{}_obj_{}.csv".format(50, 5, 3), index=False)
state2 = random_match()
random_obj = env_instance.cal_obj3(state2)
result = pd.DataFrame()
result['best_obj'] = random_obj
result['cur_obj'] = random_obj
result.to_csv("data/random_task_{}_machine_{}_obj_{}.csv".format(50, 5, 3), index=False)
print("avg obj1:", avg_obj[0])
print("random obj:", random_obj[0])
