from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import time
import core as core
from env import Env, World
from torch.utils.tensorboard import SummaryWriter

np.random.seed(2023)
writer = SummaryWriter("./logs/ddpg/t40_m5_obj3")


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def ddpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=1000, epochs=100, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn, env_fn
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    def update(data):
        loss = {}
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        loss['loss_q'] = loss_q.data.numpy()
        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()
        loss['loss_pi'] = loss_pi.data.numpy()
        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        return loss

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    episode_step = 1
    episode_num = 0
    train_num = 0
    s_list = []
    a_list = []
    next_s_list = []
    best_obj_list = []
    cur_obj_list = []
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        if t % 1000 == 0:
            print("t:", t)
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1
        episode_step += 1
        s_list.append(o)
        a_list.append(a)
        next_s_list.append(o2)
        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            for i in range(len(s_list)):
                replay_buffer.store(s_list[i], a_list[i], r, next_s_list[i], d)
            o, ep_ret, ep_len = env.reset(), 0, 0
            s_list = []
            a_list = []
            next_s_list = []
            episode_step = 1
            writer.add_scalar("reward/train", r, episode_num)
            if 'obj' in info:
                if env.best_obj is None:
                    env.best_obj = info['obj']
                elif env.best_obj > info['obj']:
                    env.best_obj = info['obj']
            writer.add_scalar("obj/train", env.best_obj, episode_num)
            best_obj_list.append(env.best_obj)
            cur_obj_list.append(info['obj'])
            episode_num += 1

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                loss_info = update(data=batch)
                writer.add_scalar("train/loss_q", loss_info['loss_q'], train_num)
                writer.add_scalar("train/loss_pi", loss_info['loss_pi'], train_num)
                train_num += 1

    return best_obj_list, cur_obj_list


if __name__ == '__main__':
    import argparse
    import pandas as pd

    # Set Network parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    # generate the environment
    world = World(task_num=40, machine_num=5)
    world.data_generator()

    env = Env(world)
    best_obj, cur_obj = ddpg(env, actor_critic=core.MLPActorCritic,
                             ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                             gamma=args.gamma, seed=args.seed, epochs=args.epochs)
    result = pd.DataFrame()
    result['best_obj'] = best_obj
    result['cur_obj'] = cur_obj
    result.to_csv("data/ddpg_task_{}_machine_{}_obj_{}.csv".format(40, 5, 3), index=False)