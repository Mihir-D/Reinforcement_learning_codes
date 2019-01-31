# This Program implements 10 armed bandit problem
# The Bandit() function takes in q, rewards, Exploration rate and no of iterations,
# and return array which contains max q value at each step.
## You can print either AVERAGE REWARD  or PERCENTAGE OF BEST ACTION

import numpy as np
import matplotlib.pyplot as plt


##############################
# q = Value function
# reward = reward of each state
# e = Exploration rate
# n = no of iterations
# returns values representing learning at each step, which can be plotted.
###############################
def Bandit(q, reward, e, n):
    count = np.zeros(len(q))
    new_reward = 0
    avg_reward = 0
    plot_reward = []
    plot_action = []
    best_action = np.argmax(reward, axis=0)
    avg_best_action = 0
    best_action_count = 0
    for i in range(n):
        action = next_action(q, e)
        if action == best_action[0]:
            best_action_count += 1
        avg_best_action = (avg_best_action*i + best_action_count)/(i+1)
        plot_action.append(avg_best_action)
        #print action
        new_reward = np.random.normal(loc=reward[action][0], scale=reward[action][1])
        avg_reward = (avg_reward*i + new_reward)/(i+1)
        plot_reward.append(avg_reward)
        count[action] += 1
        q[action] = ((count[action] - 1) * q[action] + new_reward) / count[action]  # update step
    return plot_reward, plot_action

######################################
# The next_action() function choses the best action from current q values.  
######################################  
def next_action(q, e):
    p = np.random.uniform(0,1)
    first = True
    for i in range(len(q)):
        if q[i] != 0:
            first = False
            break
    if p < e or first:
        return np.random.randint(0,len(q))
    else:
        index = 0
        max = q[0]
        for i in range(len(q)):
            if q[i]>max:
                max = q[i]
                index = i
        return index

def plot_action_percentage(plot_values1, plot_values2, plot_values3):
    plt.subplot(211)
    plt.plot(plot_values1, 'r', plot_values2, 'g', plot_values3, 'b')
    plt.ylabel("Average Action %")
    

def plot_avg_reward(plot_values1, plot_values2, plot_values3):
    plt.subplot(211)
    plt.plot(plot_values1, 'r', plot_values2, 'g', plot_values3, 'b')
    plt.ylabel("Average reward")
    

    

q = [0, 0, 0]
action_reward = [[0.2,2], [-0.3,2], [0.8,2]] # mu and sigma
e1 = 0.0 
e2 = 0.01
e3 = 0.1
n = 1000 # no of iterations

plot_values1, plot_values1a = Bandit(q, action_reward, e1, n) # plotted RED
plot_values2, plot_values2a = Bandit(q, action_reward, e2, n) # plotted GREEN
plot_values3, plot_values3a = Bandit(q, action_reward, e3, n) # plotted BLUE
plot_reward = [item[0] for item in action_reward]

plot_action_percentage(plot_values1a, plot_values2a, plot_values3a)
#plot_avg_reward(plot_values1, plot_values2, plot_values3)

plt.subplot(212)
plt.plot(plot_reward, 'bo')
plt.xlim(-1, len(q))
plt.ylim(-5,5)
plt.ylabel("reward mean")
plt.show()

# plt.subplot(211)
# plt.plot(plot_values1, 'r', plot_values2, 'g', plot_values3, 'b')
# plt.ylabel("Average reward")
# plt.subplot(212)
# plt.plot(plot_reward, 'bo')
# plt.xlim(-1, len(q))
# plt.ylim(-5,5)
# plt.ylabel("reward mean")
# plt.show()