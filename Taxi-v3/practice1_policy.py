import gym
import gym_maze
import time
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, state_n, action_n, policy_param):
        self.state_n = state_n
        self.action_n = action_n
        self.policy_param = policy_param
        self.policy = np.ones((state_n, action_n)) / action_n

    def get_action(self, state):
        prob = self.policy[state]
        action = np.random.choice(np.arange(self.action_n), p=prob)
        return int(action)

    def update_policy(self, elite_sessions):
        new_policy = np.zeros((self.state_n, self.action_n))

        for session in elite_sessions:
            for state, action in zip(session['states'], session['actions']):
                new_policy[state][action] += 1
                
        for state in range(self.state_n):
            if sum(new_policy[state]) == 0:
                new_policy[state] += 1 / self.action_n
            else:
                new_policy[state] /= sum(new_policy[state])
                
        self.policy = self.policy_param * new_policy + (1 - self.policy_param) * self.policy 

        return None


def get_session(env, agent, session_len, visual=False):
    session = {}
    states, actions = [], []
    total_reward = 0

    obs = env.reset()
    
    for _ in range(session_len):

        state = obs
        states.append(state)
        
        action = agent.get_action(state)
        actions.append(action)

        #if visual:
            #env.render()
            #time.sleep(1)


        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            if visual:
                print("total_reward ", total_reward)
            break

    session['states'] = states
    session['actions'] = actions
    session['total_reward'] = total_reward
    return session


def get_elite_sessions(sessions, q_param):

    total_rewards = np.array([session['total_reward'] for session in sessions])
    quantile = np.quantile(total_rewards, q_param)

    elite_sessions = []
    for session in sessions:
        if session['total_reward'] > quantile:
            elite_sessions.append(session)

    return elite_sessions

def learn(policy_param=0.2, q_param=0.1, episode_n=300, session_n=300, session_len=200):
    mean_rewards = []
    env = gym.make('Taxi-v3')

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = Agent(n_states, n_actions, policy_param)

    for episode in range(episode_n):
        sessions = [get_session(env, agent, session_len) for _ in range(session_n)]

        # Только для вывода в консоль. В алгоритме не используется
        mean_total_reward = np.mean([session['total_reward'] for session in sessions])
        #print('mean_total_reward = ', mean_total_reward)
        mean_rewards.append(mean_total_reward);

        elite_sessions = get_elite_sessions(sessions, q_param)

        if len(elite_sessions) > 0:
            agent.update_policy(elite_sessions)

    get_session(env, agent, session_len, visual=True)
    plt.plot([i for i in range(episode_n)], mean_rewards, 'g')
    plt.axis([0, episode_n, -20, 20])
    plt.xlabel('Episode number')
    plt.ylabel('Mean reward')
    plt.title(f'Policy coefitient {policy_param}')
    plt.show()


learn()

