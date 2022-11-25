##### TODO:

1. come up with two interesting MDPs, explain why interesting; keep it simple, max one grid world; vary the number of states
2. solve each MDP with both value iteration and policy iteration;
3. describe iterations and time?
3. describe how convergence is measured?
4. do they reach the same conclusions?
5. how did the number of states affect it all?
6. solve the two MDPs with a different RL approach... one that is not model-based
7. did the new approach perform better than the model based approaches?

#### MDP Problems

##### Problem 1: Forest Management

This problem involves balancing the tradeoffs in managing a forest and is found in the examples module of the [pymdptoolbox](https://pymdptoolbox.readthedocs.io/en/latest/api/example.html#mdptoolbox.example.forest) python package. The agent aims to *(a)* maintain an old forest ecosystem for wildlife, and *(b)* profit from logging. In the "game" the agent has two actions: {*Wait*, *Cut*} and this action applies to the entire forest (not tree by tree). Each state is an age of the forest. The rewards work as follows: if the forest is Cut when the forest is in the oldest state then the reward is 2, if the forest is not cut in the oldest state then the reward is 4. As well, when the forest is cut in any other state than the oldest state and the youngest state the reward is 1. The tricky part of the problem is that in each period there is some probability that the forest will burn down, in which case the reward is 0.

This problem is interesting because the number of actions is fixed at two while states can increase infinitely. However, the probability of reaching the oldest forest state approaches zero as the number of states increases. So this problem likely only has a solution for a "reasonable" number of states, to be determined how many states constitutes reasonable. Interestingly, the idea of a "best state" here doesn't apply as obviously the oldest state is optimal, the important question involves the optimal policy to weigh waiting for that state and the potential risk of fire.

##### Problem 2: Random Markov Process

This problem represents generating a random transition probability matrix, and a random reward matrix. It is found in the examples module of [pymdptoolbox](https://pymdptoolbox.readthedocs.io/en/latest/api/example.html#mdptoolbox.example.rand) as well. Because these matrixes are entirely randomized, it's hard to imagine that an optimal policy exists that well-outperforms other policies.

#### Value Iteration

Value iteration convergence is reached when the change in the value function between iterations is less than 0.01. If this condition is not met, then the algorithm will terminate after 1000 iterations (at least for my implementation).

##### Problem 1

Because of the simple environment for this problem, convergence was reached for all number of states tried, as well as three different discount factors (0.9, 0.95, 0.99). Unsurprisingly, more iterations are required for the value function to converge, given the same number of states. This is because the future is more valuable when the discount rate is higher, and as such, more of it need to be explored. Interesting to note here is the exponential growth in the number of iterations as a function of number of states: between2 and roughly 10 states the number of iterations grows linearly, however beyond that point (until convergence) the number of iterations explodes. This shows the limitations of value iteration in that it requires (at least in the limit) to visit each state infinitely often to converge to a global optimum, obviously as more states are added this becomes more difficult. Still, this problem is relatively easy to solve because there are only two actions regardless of the number of states.

![](plots/Problem_1_VI_iterations_by_states.png)

This second plot that looks at time to converge by number of states should be taken with a grain of salt since the time represents just a single sample. However, the relationship between time and states does not appear to be as exponential in number of states as does the relationship between iterations and state. The 0.99 discount factor curve does exhibit an explosion at roughly state 18, however before then it's quite linear. This is likely because each iteration is relatively cheap and still there are not a huge number of states.

![](plots/Problem_1_VI_time_by_states.png)

This figure requires some explanation, but it compares how the value function changes over the each state, comparing across different state lengths. The idea is that the best state is obviously the max age, because then we can either wait for reward of 4 or cut for a reward of 2. The question becomes, how quickly does the value function's value drop as we move away from the optimal state/value pair?

Thinking about the simplest case with only two states: *cleared forest* and *fully grown forest*, the curve is a straight line. As we add more states, transitionary states become available between cleared and fully grown and as such the line turns into more of a curve. This is because of the dynamics of our world. For example, state 5 is very valuable if there are only 5 states while it is not very valuable if there are 50 states because the probability of reaching state 50 before a fire approaches zero. The result is that when the number of total states gets large, there is no attempt to reach the oldest state but rather just cut down whenever possible. This is shown by the lark lines that refer to larger total number of states, where the value function values hover at around 0.3 in early states. Meanwhile, for smaller total number of states it is possible to reach the oldest state so the new growth states hover around 0.2, accepting waiting early on in order to reach the final state.

![](plots/Problem_1_VI_distribution_of_value_functions.png)

##### Problem 2

All run with discount rate of 0.95.

In the second MDP problem, number of iterations appears to be more a function of the number of actions as opposed to the number of states. In this figure we see that the largest number of iterations is actually spent on problems that have few states but many actions. This is surprising to me, but I can rationalize it with the following: perhaps with few states and a lot of actions the value functions are relatively constant over iterations as the difference of being in state 1 vs state 2 is overshadowed by the difference in choosing some action in either state. I also thought about whether or not the learning rate, epsilon, might need to be adjusted in response to different MDPs. Maybe as the world gets more complicated it's easier to stumble into some region in which the improvements fall below the epsilon threshold. As well, because the transition matrix and reward matrix are chosen at random, it could be the case that the distribution used gives way to some outlier action-state combinations that are clearly optimal. Unlikely yes, but I suppose possible.

![](plots/Problem_2_VI_iterations_by_states.png)

In this figure, we see that at low number of actions, the relationship between states and time is somewhat linear, but as actions are increased this relationship becomes more and more convex. Especially the problem with the largest number of actions, 128, the time curve just shoots upwards almost immediately. What I'm having a hard time wrapping my head around is why medium actions and medium states would take so much less time than many actions and few states. Both combinations represent varying levels of world complexity, but it appears actions carry more importance than states (at least for this narrowly defined problem).

![](plots/Problem_2_VI_time_by_states.png)

Considering the figures above, an explanation could be that the solutions are simply not good, despite the fact that convergence has been reached. On the one hand I understand value iteration to reach a global optimum, but on the other hand it requires visiting each state infinitely often and for larger problems this isn't possible (especially on my MacBook). That said, looking at the solutions to the various MDP state-action combinations, all combinations converge to the epsilon optimum policy.

#### Policy Iteration

##### Problem 1

Policy iteration requires fewer iterations than value iteration to solve problem 1, across all states and discount factors. As well, the number of iterations is linear in the number of states rather than exponential as seen in value iteration.

![](plots/Problem_1_PI_iterations_by_states.png)

In the time domain however, policy iteration is noticeably slower (at times exceeding double the time) than value iteration. This is likely because of the nested iterative approach of policy iteration, alternating between exploration of new policies and exploitation of known policies. Interestingly, the time jumps most at smaller number of states, and levels off at about 9 states. This is the opposite relationship to value iteration which only slowed down noticeably at a higher number of states. Clearly, at the beginning policy iteration benefits greatly from exploring several policies and pursuing those that are fruitful.

![](plots/Problem_1_PI_time_by_states.png)

Looking at how the value function's values changes as we move from the oldest state, we see that policy iteration does not discount as much as we move from the optimal/oldest state. Why might that be? I think this is because policy iteration doesn't care about the values of value function per se. All it cares about is the relative ordering of the policies. This is akin to using regression probabilities for a classification problem, we choose the class that has the highest probability without caring about the specific value of the probability.

![](plots/Problem_1_PI_distribution_of_value_functions.png)

I tried to verify this by comparing the policies across value and policy iteration and they are identical.

##### Problem 2

Policy iteration seems to have no problem reaching optimal solutions in few iterations. Even in the situations that gave value iteration trouble: large number of actions and few states policy iteration converges in less than 8 iterations for all combinations tested. This could be a quirk of the randomization, and in practice one should repeat these experiments with a variety of starting positions and transition/reward matrices.

![](plots/Problem_2_PI_iterations_by_states.png)

The tradeoff for fewer number of iterations for policy iteration convergence is that more computation is done in each iteration, making them take longer. We see this in the following figure, where the MDP with the largest number of states takes close to a minute to find an optimal solution.

![](plots/Problem_2_PI_time_by_states.png)

Interestingly, in this problem there is a setup that results in value iteration and policy iteration arriving at different conclusions. When there are 32 states and 2 actions the two solvers are not in line. Since this setup with 32 states and 2 actions is relatively small, this gives reason to believe that other situations like this are possible. That said, the two solvers do agree on the best state and also on which policy to take in the best state. The correlation between the value function is also equal to 1. To me, this goes to show that in a complicated enough environment as a large MDP it could be the case that there are multiple policies that are very comparable in terms of overall utility.

#### Another RL Approach

##### Problem 1

TBD

##### Problem 2

TBD

#### Conclusion