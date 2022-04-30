import argparse
import os
from typing import List, Optional, Tuple

import gym
import numpy as np
import pettingzoo.butterfly.pistonball_v6 as pistonball_v6
import torch
from tianshou.data import VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from torch.utils.tensorboard import SummaryWriter

from marl_comm.data import MACollector, MAReplayBuffer
from marl_comm.env import MAEnvWrapper, get_MA_VectorEnv
from marl_comm.ma_policy import MAPPOPolicy, PPOPolicy


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma",
                        type=float,
                        default=0.9,
                        help="a smaller gamma favors earlier win")
    parser.add_argument("--n-pistons",
                        type=int,
                        default=10,
                        help="Number of pistons(agents) in the env")
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--step-per-epoch', type=int, default=500)
    parser.add_argument('--step-per-collect', type=int, default=100)
    parser.add_argument('--repeat-per-collect', type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--hidden-sizes",
                        type=int,
                        nargs="*",
                        default=[64, 64])
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument('--rew-norm', type=int, default=False)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--lr-decay', type=int, default=True)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument("--joint-critic", action="store_true")
    parser.add_argument("--render", type=float, default=0.)

    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, "
        "watch the play of pre-trained models",
    )
    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env(args: argparse.Namespace = get_args()):
    return MAEnvWrapper(
        pistonball_v6.env(continuous=False, n_pistons=args.n_pistons))


def get_agents(
    args: argparse.Namespace = get_args(),
) -> Tuple[BasePolicy, List[torch.optim.Optimizer], List]:

    env = get_env()
    observation_space = (env.observation_space["observation"] if isinstance(
        env.observation_space, gym.spaces.Dict) else env.observation_space)
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    agents = []
    for _ in range(args.n_pistons):
        # model
        net_a = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)
        actor = Actor(net_a,
                      args.action_shape,
                      device=args.device,
                      softmax_output=False)
        net_c = Net(np.prod(args.state_shape) + np.prod(env.state_space.shape)
                    if args.joint_critic else np.prod(args.state_shape),
                    hidden_sizes=args.hidden_sizes,
                    device=args.device)
        critic = Critic(net_c, device=args.device)
        optim = torch.optim.Adam(set(actor.parameters()).union(
            critic.parameters()),
                                 lr=args.lr)

        def dist(p):
            return torch.distributions.Categorical(logits=p)

        agent = PPOPolicy(actor,
                          critic,
                          optim,
                          dist,
                          joint_critic=args.joint_critic,
                          discount_factor=args.gamma,
                          gae_lambda=args.gae_lambda,
                          max_grad_norm=args.max_grad_norm,
                          vf_coef=args.vf_coef,
                          ent_coef=args.ent_coef,
                          reward_normalization=args.rew_norm,
                          action_scaling=False,
                          action_space=env.action_space,
                          eps_clip=args.eps_clip,
                          value_clip=args.value_clip,
                          dual_clip=args.dual_clip,
                          advantage_normalization=args.norm_adv,
                          recompute_advantage=args.recompute_adv).to(
                              args.device)
        agents.append(agent)
    policy = MAPPOPolicy(
        agents,
        env,
        joint_critic=args.joint_critic,
    )
    return policy, env.agents


def get_buffer(args: argparse.Namespace = get_args()):
    env = get_env()
    return MAReplayBuffer(args.buffer_size, env.agents, VectorReplayBuffer,
                          args.training_num)


def train_agent(
        args: argparse.Namespace = get_args(), ) -> Tuple[dict, BasePolicy]:
    train_envs = get_MA_VectorEnv(SubprocVectorEnv,
                                  [get_env for _ in range(args.training_num)])
    test_envs = get_MA_VectorEnv(SubprocVectorEnv,
                                 [get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    policy, agents = get_agents(args)

    # collector
    train_collector = MACollector(policy,
                                  train_envs,
                                  get_buffer(args),
                                  exploration_noise=True)
    test_collector = MACollector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, "pistonball", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        pass

    def stop_fn(mean_rewards):
        return False

    def reward_metric(rews):
        return rews[0]

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    return result, policy


def watch(args: argparse.Namespace = get_args(),
          policy: Optional[BasePolicy] = None) -> None:
    env = get_MA_VectorEnv(DummyVectorEnv, [get_env])
    policy.eval()
    collector = MACollector(policy, env)
    result = collector.collect(n_episode=1, render=args.render)
    # print(collector.data)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[0].mean()}, length: {lens.mean()}")


def test_piston_ball(args=get_args()):
    import pprint

    if args.watch:
        watch(args)
        return

    result, agent = train_agent(args)
    # assert result["best_reward"] >= args.win_rate

    if __name__ == "__main__":
        pprint.pprint(result)
        # Let's watch its performance!
        watch(args, agent)


if __name__ == "__main__":
    test_piston_ball(get_args())
