import chex
import jax
import mctx
from jax import numpy as jnp

from matplotlib import pyplot as plt

import window_method
from dfn_env import ParaState, Paratest, reward
import joblib


def policy_function(embedding: ParaState) -> chex.Array:
    return jnp.array([1, 1.0, 1, 1, 1, 1])


def value_function(embedding: ParaState) -> chex.Array:
    return embedding.reward


def root_fn(embedding: ParaState, rng_key: chex.PRNGKey) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits=policy_function(embedding),
        value=value_function(embedding),
        embedding=embedding,
    )


def invalid_actions(embedding: ParaState, rng_key: chex.PRNGKey) -> chex.Array:
    return ~embedding.legal_action_mask


def recurrent_fn(Params, rng_keys, action: int, embedding: ParaState):
    embedding = env.step(embedding, action.astype(int))
    reward = embedding.reward
    value = value_function(embedding)
    # value = jnp.where(embedding.terminated, 100.0, value)
    discount = 0.5 * jnp.ones_like(value)
    # discount = jnp.where(embedding.terminated, 1.0, discount).astype(jnp.float32)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=policy_function(embedding),
        value=value,
    )
    return recurrent_fn_output, embedding


# @functools.partial(jax.jit, static_argnums=(2,))
def pure_mcts(rng_keys: chex.PRNGKey, embedding: ParaState, num_simulations: int, depth: int) -> chex.Array:
    batch_size = 1
    key1, key2 = jax.random.split(rng_keys)
    policy_output = mctx.gumbel_muzero_policy(
        params=None,
        rng_key=key1,
        root=jax.vmap(root_fn, (None, 0))(embedding, jax.random.split(key2, batch_size)),
        recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
        num_simulations=num_simulations,
        max_depth=depth,
        qtransform=mctx.qtransform_completed_by_mix_value,
        invalid_actions=jax.vmap(invalid_actions,(None, 0))(embedding, jax.random.split(key2, batch_size)),
        gumbel_scale=0.3,
    )
    return policy_output


def env_test(embedding, env, action, key) -> ParaState:
    # print(embedding.parameters_test[0,])
    embedding = jax.vmap(env.step)(embedding, action.astype(int))

    # print(embedding.parameters_test[0,])
    # # print(a.parameters_target)
    # print(embedding.reward[0,])
    # print(embedding.terminated[0,])
    # print(f"value_function(a){value_function(embedding)}")
    # print(f"policy_function(a){policy_function(embedding)}")
    print(f"root_fn{recurrent_fn(embedding=embedding, action=action)[1]}")
    return embedding


if __name__ == "__main__":
    _, lines = window_method.load_lines('points.csv')
    scale = 0.14/360
    batch_num = 10
    target = lines * scale
    env = Paratest(target=target, observation_area=jnp.array(0.5))
    keys = jax.random.PRNGKey(2)
    random_keys = jax.random.split(keys, num=batch_num)
    total_reward, total_p21, total_kl = [], [], []
    for i, subkey in enumerate(random_keys):
        record_r, record_p21, record_kl = [], [], []
        a = env.reset(rng_key=subkey, test_parameters=jnp.array([0.8, 1.0]))
        print(f"group{i}start")
        while ~a.terminated:
            action = pure_mcts(keys, embedding=a, num_simulations=10000, depth=5).action_weights
            action = action.argmax().item()
            a = env.step(a, action)
            rewardlist = []
            record_r.append(a.reward)
            record_p21.append(a.info[1])
            record_kl.append(a.info[0])
            length = len(record_r)
            if length > 6:
                last_n = record_r[-min(length, 10):]
                if last_n.index(max(last_n)) == 0:
                    break
        total_reward.append(record_r)
        total_p21.append(record_p21)
        total_kl.append(record_kl)
        a.result_plt(save=True)
        print(f"group{i}'sDE{a.info[0]:.4f},IE:{a.info[1]:.4f},reward:{a.reward:.4f}"
              f"length is {a.parameters_test[1]*0.03}m \ndensity is {a.parameters_test[0]*500}/m3")
        plt.close('all')
    fixed_length = 15
    processed_data = []
    for rewards in total_reward:
        effective_length = min(len(rewards) - 5, fixed_length)
        if len(rewards) > fixed_length:
            processed_rewards = rewards[:fixed_length]
        else:
            processed_rewards = rewards[:effective_length] + [jnp.nan] * (fixed_length - effective_length)
        processed_data.append(processed_rewards)

    processed_data = jnp.array(processed_data)
    data = {"total_reward": total_reward, "total_p21": total_p21,"total_kl": total_kl}
    joblib.dump(data, "data_log.pkl")

    average_rewards = jnp.nanmean(processed_data, axis=0)
    max_rewards = jnp.nanmax(processed_data, axis=0)
    min_rewards = jnp.nanmin(processed_data, axis=0)
    actions = jnp.arange(1, fixed_length + 1)
    plt.plot(actions, average_rewards, label="Average Reward", color="green")
    plt.fill_between(actions,
                     min_rewards,
                     max_rewards,
                     color="green", alpha=0.1, label="Reward Range")
    plt.xlabel("Action step")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
