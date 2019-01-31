# This Program implements 10 armed bandit problem
# The Bandit() function takes in q, rewards, Exploration rate and no of iterations,
# and return array which contains max q value at each step.


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
    plot_values = []
    for i in range(n):
        action = next_action(q, e)
        #print action
        r = np.random.normal(loc=reward[action][0], scale=reward[action][1])
        #print r
        count[action] += 1
        q[action] = ((count[action] - 1) * q[action] + r) / count[action]  # update step
        plot_values.append(max(q))
    return plot_values

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

q = [0, 0, 0]
reward = [[1,2], [5,2], [-1,2]]
e1 = 0.0 
e2 = 0.01
e3 = 0.1
n = 1000 # no of iterations

plot_values1 = Bandit(q, reward, e1, n)
plot_values2 = Bandit(q, reward, e2, n)
plot_values3 = Bandit(q, reward, e3, n)


plt.plot(plot_values1)
plt.plot(plot_values2)
plt.plot(plot_values3)
plt.ylabel("q_values")
plt.show()