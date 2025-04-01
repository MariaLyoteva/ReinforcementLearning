"""
Small demo to illustrate how the plot function and the gridworld environment work
"""
import numpy as np 
import time 
from gridworld import *
from plot import *
from config import G_UP, G_RIGHT, G_DOWN, G_LEFT

ACTIONS = [G_UP, G_RIGHT, G_DOWN, G_LEFT]
def value_iteration(env, gamma=0.7, theta=1e-4, in_place=True):
    """
    Value iteration algorithm (supports in-place updates).
    :param env: Gridworld environment
    :param gamma: Discount factor
    :param theta: Convergence threshold
    :param in_place: Whether to use in-place dynamic programming
    :return: v_table, policy
    """
    num_states = env.num_states()
    num_actions = env.num_actions()
    v_table = np.zeros(num_states)

    while True:
        delta = 0
        old_v_table = v_table.copy()

        for s in range(num_states):
            q_values = []
            for a in ACTIONS:
                s_next, reward, done = env.step_dp(s, a)
                v_next = v_table[s_next] if in_place else old_v_table[s_next]
                q = reward + gamma * v_next * (not done)
                q_values.append(q)
            
            max_q = max(q_values)
            delta = max(delta, abs(v_table[s] - max_q))
            v_table[s] = max_q

        if delta < theta:
            break

    # Derive greedy policy
    policy = np.zeros((num_states, num_actions))
    for s in range(num_states):
        q_values = []
        for a in ACTIONS:
            s_next, reward, done = env.step_dp(s, a)
            q = reward + gamma * v_table[s_next] * (not done)
            q_values.append(q)

        best_action = np.argmax(q_values)
        policy[s][best_action] = 1

    return v_table, policy

def policy_iteration(env, gamma=0.99, theta=1e-4):
    num_states = env.num_states()
    num_actions = env.num_actions()

    # Initialize policy randomly (uniformly)
    policy = np.ones((num_states, num_actions)) / num_actions
    v_table = np.zeros(num_states)

    while True:
        # --- Policy Evaluation ---
        while True:
            delta = 0
            for s in range(num_states):
                v = 0
                for a in range(num_actions):
                    s_next, reward, done = env.step_dp(s, a)
                    v += policy[s][a] * (reward + gamma * v_table[s_next] * (not done))
                delta = max(delta, abs(v - v_table[s]))
                v_table[s] = v
            if delta < theta:
                break

        # --- Policy Improvement ---
        stable = True
        for s in range(num_states):
            old_action = np.argmax(policy[s])
            q_values = []
            for a in range(num_actions):
                s_next, reward, done = env.step_dp(s, a)
                q = reward + gamma * v_table[s_next] * (not done)
                q_values.append(q)

            best_action = np.argmax(q_values)
            policy[s] = np.eye(num_actions)[best_action]

            if best_action != old_action:
                stable = False

        if stable:
            break

    return v_table, policy



def derive_q_table(env, v_table, gamma=0.7):
    num_states = env.num_states()
    num_actions = env.num_actions()
    q_table = np.zeros((num_states, num_actions))

    for s in range(num_states):
        for a in ACTIONS:
            s_next, reward, done = env.step_dp(s, a)
            q_table[s, a] = reward + gamma * v_table[s_next] * (not done)

    return q_table

if __name__ == "__main__":
    # create environment
    env = ExerciseWorld()
    cenv = CustomWorld()
    # create nonsense V-values and nonsense policy
    #v_table = np.random.random((env.num_states()))
    #q_table = np.random.random((env.num_states(), env.num_actions()))
    #policy = np.random.random((env.num_states(), env.num_actions()))
    ##v_table, policy = value_iteration(env, in_place=True)
    ##q_table = derive_q_table(env, v_table)
    # either plot V-values and Q-values without the policy...
    # plot_v_table(env, v_table)
    # plot_q_table(env, q_table)
    # ...or with the policy
    ##plot_v_table(env, v_table, policy)
    ##plot_q_table(env, q_table, policy)
    
    start = time.time()
    #v, p = value_iteration(env, in_place=True)
    v_pi, pi_pi = policy_iteration(env)
    print("Policy:", time.time() - start)

    start = time.time()
    v2, p2 = value_iteration(env, in_place=True)
    print("Value in-place took:", time.time() - start)
