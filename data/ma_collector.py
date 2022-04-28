import time
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np
import torch
from marl_comm.env import get_MA_VectorEnv
from tianshou.data import AsyncCollector, Batch, to_numpy
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv, DummyVectorEnv


class MultiAgentCollector(AsyncCollector):
    def __init__(
        self,
        env: Union[gym.Env, BaseVectorEnv],
        **kwargs: Any,
    ) -> None:
        if hasattr(env, "num_agents"):
            agents = env.agents
        else:
            agents = env.get_env_attr("agents", [0])[0]
        if hasattr(env, "agent_idx"):
            agent_idx = env.agent_idx
        else:
            agent_idx = env.get_env_attr("agents_idx", [0])[0]

        self.agents = agents
        self.agent_idx = agent_idx
        self.agent_num = len(agent_idx)

        if not isinstance(env, BaseVectorEnv):
            env = get_MA_VectorEnv(DummyVectorEnv, [lambda: env])

        self.maenv_num = env.env_num

        super().__init__(env=env, **kwargs)

    def reset_env(self) -> None:
        """Reset all of the environments.
        obs is inited in the following format:
        [{'agent_id': 'agent_id_for_agent0', 'obs': obs0}, {'agent_id': 'agent_id_for_agent1', 'obs': empty_array}, ...]
        """
        local_obs = self.env.reset()

        self._ready_env_ids = np.array(
            [
                self.agent_idx[local_obs[env_i]["agent_id"]] * len(local_obs) + env_i
                for env_i in range(len(local_obs))
            ]
        )
        global_obs = []
        for agent_i in range(0, self.agent_num):
            for _ in range(self.env.env_num):  # self.env.env_num is the num of MAEnvs
                item = {
                    "agent_id": self.agents[agent_i],
                    "obs": np.empty_like(local_obs[0]["obs"]),
                }
                if "mask" in local_obs[0]:
                    item["mask"] = [False for _ in range(len(local_obs[0]["mask"]))]
                global_obs.append(item)

        for local_env_i, obs in enumerate(local_obs):
            global_env_i = (
                self.agent_idx[obs["agent_id"]] * len(local_obs) + local_env_i
            )
            global_obs[global_env_i] = obs

        obs = global_obs

        if self.preprocess_fn:
            obs = self.preprocess_fn(obs=global_obs, env_id=self._ready_env_ids).get(
                "obs", global_obs
            )
        self.data.obs = obs

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
    ) -> Dict[str, Any]:
        # collect at least n_step or n_episode
        if n_step is not None:
            assert n_episode is None, (
                "Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
        elif n_episode is not None:
            assert n_episode > 0
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        ready_env_ids = self._ready_env_ids

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        while True:
            whole_data = self.data
            self.data = self.data[ready_env_ids]

            assert len(whole_data) == self.env_num  # major difference
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                try:
                    act_sample = [
                        self._action_space[i % self.maenv_num].sample()
                        for i in ready_env_ids
                    ]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            result = self.env.step(action_remap, ready_env_ids)  # type: ignore
            obs_next, rew, done, info = result

            _rew = np.take_along_axis(
                rew, np.expand_dims(ready_env_ids, -1) // self.maenv_num, -1
            ).reshape(-1)

            rew = np.array(rew).transpose(1, 0).reshape(-1)

            self.data.update(obs_next=obs_next, done=done, info=info, rew=_rew)

            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        env_id=ready_env_ids,
                    )
                )
                rew = self.preprocess_fn(rew=rew, env_id=ready_env_ids)

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )

            try:
                whole_data.act[ready_env_ids] = self.data.act
                whole_data.policy[ready_env_ids] = self.data.policy
                whole_data.obs_next[ready_env_ids] = self.data.obs_next
                whole_data.done[ready_env_ids] = self.data.done
                whole_data.info[ready_env_ids] = self.data.info

                whole_data.rew = rew
            except ValueError:
                _alloc_by_keys_diff(whole_data, self.data, self.env_num, False)
                whole_data[ready_env_ids] = self.data  # lots of overhead

            # collect statistics
            step_count += len(ready_env_ids)

            # major differnece from the async case
            # change self.data here because ready_env_ids has changed
            try:
                ready_env_ids = info["env_id"]
            except Exception:
                ready_env_ids = np.array([i["env_id"] for i in info])

            last_data = self.data
            self.data = whole_data[ready_env_ids]

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                obs_reset = self.env.reset(env_ind_global)
                if self.preprocess_fn:
                    obs_reset = self.preprocess_fn(
                        obs=obs_reset, env_id=env_ind_global
                    ).get("obs", obs_reset)
                last_data.obs_next[env_ind_local] = obs_reset
                for i in env_ind_local:
                    self._reset_state(i)

            try:
                whole_data.obs[ready_env_ids] = last_data.obs_next
            except ValueError:
                _alloc_by_keys_diff(whole_data, self.data, self.env_num, False)
                self.data.obs = last_data.obs_next
                whole_data[ready_env_ids] = self.data  # lots of overhead
            self.data = whole_data

            if (n_step and step_count >= n_step) or (
                n_episode and episode_count >= n_episode
            ):
                break

        self._ready_env_ids = ready_env_ids

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if episode_count > 0:
            rews, lens, idxs = list(
                map(np.concatenate, [episode_rews, episode_lens, episode_start_indices])
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
        }
