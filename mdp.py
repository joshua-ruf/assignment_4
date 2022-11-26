# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import seaborn as sns
import mdptoolbox
import mdptoolbox.example
import time
import gc
from tqdm import tqdm

from itertools import product
from pprint import pprint


# -

def plot_and_save(title, kind=sns.lineplot, xlabel=None, ylabel=None, **kwargs):
    plot = kind(**kwargs)
    plot.set_title(title.upper())

    if xlabel:
        plot.set_xlabel(xlabel)
        
    if ylabel:
        plot.set_ylabel(ylabel)
    
    fig = plot.get_figure()
    
    file = f"plots/{title.replace(' ', '_')}.png"
    fig.savefig(file)
    
    print(file)

# +
# # forest example
# P, R = mdptoolbox.example.forest(2)
# mod = mdptoolbox.mdp.QLearning(P, R, 0.95)
# mod.run()

# getattr(mod, 'V', None)

# pprint(mod.Q)

# # del mod
# # gc.collect()

# +
# P, R = mdptoolbox.example.rand(2, 3)

# P.shape, R.shape
# -



# +
def make_P_and_R(states, actions=None, example='forest'):
    if example == 'forest':
        return mdptoolbox.example.forest(states)
    else:
        np.random.seed(0)
        return mdptoolbox.example.rand(states, actions)

def run_iteration(P, R, solver=mdptoolbox.mdp.ValueIteration, **kwargs):
    mod = solver(P, R, **kwargs)
    mod.run()
    
    best_state = np.argmax(mod.V)
    return {
        'value_function': mod.V,
        'policy': mod.policy,
        'iterations': getattr(mod, 'iter', None),
        'time': mod.time,
        'states': mod.S,
        'actions': mod.A,
        'solver': solver.__name__,
        'best_state': best_state,
        'policy_in_best_state': mod.policy[best_state],
        **{f"kwargs__{v}": k for v, k in kwargs.items()},
    }

    

# +
### params
# 4096*32
params = {
    'states': np.power(2, np.arange(1, 16)),
    'actions': np.power(2, np.arange(1, 8)),
    'discounts': [0.95],
    'solvers': (mdptoolbox.mdp.ValueIteration, mdptoolbox.mdp.PolicyIteration),
    'examples': ['rand'],
}

n = len([x for x in product(*params.values())])
pprint(params)
print(f"{n} iterations")
# -

MAX = (4096**2)*32
MAX

# +
MAX = (4096**2)*32

t0 = time.time()

out = []
iterator = tqdm(product(*params.values()))
for S, A, discount, solver, example in iterator:    
    if S*S*A > MAX:
        continue
    
    run = run_iteration(*make_P_and_R(S, A, example), solver=solver, discount=discount)
    run['example'] = example
    out.append(run)
    
    del run
    gc.collect()

print(f"Done! Took {(time.time() - t0)/60:.2f} mins")

# -

df = pd.DataFrame(out)
df.to_csv('rand-data.csv', index=False)

df.actions.max(), df.states.max()

df.time.describe()

# +
### params
# states will have to be reduced for the forest management situation
params = {
    'states': np.arange(2, 30, 1),
    'actions': [None],
    'discounts': [0.9, 0.95, 0.99],
    'solvers': (mdptoolbox.mdp.ValueIteration, mdptoolbox.mdp.PolicyIteration),
    'examples': ['forest'],
}

n = len([x for x in product(*params.values())])
print(f"{n} iterations")

# +
t0 = time.time()

out_forest = []
iterator = enumerate(product(*params.values()))
for i, (S, A, discount, solver, example) in iterator:
    if i % 50 == 0:
        print(f"{i}/{n}")
    
    P, R = make_P_and_R(S, A, example)
    run = run_value_iteration(P, R, solver=solver, discount=discount)
    run['example'] = example
    out_forest.append(run)
    
print(f"Done! Took {time.time() - t0}s")

# -

"""
- note the best state doesn't mean a lot here, it's obviously the oldest state
"""
df_forest = pd.DataFrame(out_forest)

# +
# df_forest

plot_and_save('Problem 1 VI iterations by states',
              data=df_forest[df_forest.solver == 'ValueIteration'],
              x='states', y='iterations', hue='kwargs__discount')

# -

plot_and_save('Problem 1 VI time by states',
              data=df_forest[df_forest.solver == 'ValueIteration'],
              x='states', y='time', hue='kwargs__discount')


test = df_forest.value_function.apply(lambda x: [y/max(x) for y in x]).explode()
test = test.to_frame()
test['index'] = test.index
test['state'] = 1
test['state'] = test.groupby('index')['state'].cumsum()
test['state'] /= test.groupby('index')['state'].max()
test = test.join(df_forest[['solver', 'kwargs__discount', 'states']])
test

plot_and_save('Problem 1 VI distribution of value functions',
              data=test[(test.solver == 'ValueIteration') & (test.kwargs__discount == 0.95)],
              x='state', y='value_function', hue='states',
             xlabel='state (divided by number of states)',
             ylabel='value_function value (divided by max)')


# +

plot_and_save('Problem 1 PI iterations by states',
              data=df_forest[df_forest.solver == 'PolicyIteration'],
              x='states', y='iterations', hue='kwargs__discount')
# -


plot_and_save('Problem 1 PI time by states',
              data=df_forest[df_forest.solver == 'PolicyIteration'],
              x='states', y='time', hue='kwargs__discount')

plot_and_save('Problem 1 PI distribution of value functions',
              data=test[(test.solver == 'PolicyIteration') & (test.kwargs__discount == 0.95)],
              x='state', y='value_function', hue='states',
             xlabel='state (divided by number of states)',
             ylabel='value_function value (divided by max)')


# check if policies are the same for value and policy iteration
cols = ['states', 'actions', 'kwargs__discount']
df_forest.groupby(cols)['policy'].nunique().describe()


# +
##### P2 #####

# standard deviation of value_function values
df['value_fucntion_std'] = df.value_function.apply(np.std)

def f(x):
    if len(x) < 2:
        return None
    
    y = sorted(x, reverse=True)
    return y[0]/y[1]

# how many times the second best state is the best state?
df['value_function_pct_spread'] = 100*(df.value_function.apply(f) - 1)

df.head()
# -

plot_and_save('Problem 2 VI iterations by states',
              data=df[
                  (df.solver == 'ValueIteration')
              ],
              x='states', y='iterations', hue='actions')

plot_and_save('Problem 2 PI iterations by states',
              data=df[
                  (df.solver == 'PolicyIteration')
              ],
              x='states', y='iterations', hue='actions')

plot_and_save('Problem 2 VI time by states',
              data=df[(df.solver == 'ValueIteration')],
              x='states', y='time', hue='actions')

plot_and_save('Problem 2 PI time by states',
              data=df[(df.solver == 'PolicyIteration')],
              x='states', y='time', hue='actions')

# +
test = df.groupby(['actions', 'states'])['policy'].nunique()
test = test.to_frame().reset_index()
test = test[test.policy == 2]

print(test)

np.corrcoef(*pd.merge(test, df, on=['actions', 'states']).value_function.values)

# +
# plot_and_save("Problem 2 VI SD of states' values by states",
#               data=df[(df.solver == 'ValueIteration')],
#               x='states', y='value_fucntion_std', hue='actions')

# +
# plot_and_save("Problem 2 VI % gap between best and 2nd best state, by state",
#               data=df[(df.solver == 'ValueIteration')],
#               x='states', y='value_function_pct_spread', hue='actions')

# +
####### Q learning #########

# +
# forest
params = {
    'states': np.arange(2, 30, 1),
    'actions': [None],
    'discounts': [0.9, 0.95, 0.99],
    'solvers': (mdptoolbox.mdp.QLearning, mdptoolbox.mdp.ValueIteration),
    'examples': ['forest'],
}

n = len([x for x in product(*params.values())])
pprint(params)
print(f"{n} iterations")

# +
t0 = time.time()

out_forest = []
iterator = tqdm(product(*params.values()))
for S, A, discount, solver, example in iterator:
        
    params = {
        'discount': discount,
    }
    if solver.__name__ == 'QLearning':
        params['n_iter'] = 100000
    
    run = run_iteration(*make_P_and_R(S, A, example), solver=solver, **params)
    run['example'] = example
    out_forest.append(run)
    
    del run
    gc.collect()
    
print(f"Done! Took {time.time() - t0}s")
# -

df_forest = pd.DataFrame(out_forest)
df_forest.head()

plot_and_save('Problem 1 QL time by states',
              data=df_forest[df_forest.solver == 'QLearning'],
              x='states', y='time', hue='kwargs__discount')

# +
cols = ['states', 'kwargs__discount']
test = df_forest.groupby(cols)['policy'].agg(lambda x: np.equal(*x).mean())
test = test.to_frame().reset_index()

test
# -

plot_and_save('Problem 1 QL policy agreement with VI',
              data=test,
              drawstyle='steps-pre',
              x='states', y='policy', hue='kwargs__discount',
             ylabel='% agreement in policy function')



# +
# rand
params = {
    'states': np.power(2, np.arange(1, 16)),
    'actions': np.power(2, np.arange(1, 8)),
    'discounts': [0.95],
    'solvers': (mdptoolbox.mdp.ValueIteration, mdptoolbox.mdp.QLearning),
    'examples': ['rand'],
}

n = len([x for x in product(*params.values())])
pprint(params)
print(f"{n} iterations")

# +
MAX = (4096**2)*32

t0 = time.time()

out = []
iterator = tqdm(product(*params.values()))
for S, A, discount, solver, example in iterator:    
    if S*S*A > MAX:
        continue
    
    params = {
        'discount': discount,
    }
    if solver.__name__ == 'QLearning':
        params['n_iter'] = 100000
    
    run = run_iteration(*make_P_and_R(S, A, example), solver=solver, **params)
    run['example'] = example
    out.append(run)
    
    del run
    gc.collect()

print(f"Done! Took {(time.time() - t0)/60:.2f} mins")

# -

df = pd.DataFrame(out)
df.to_csv('q-data.csv', index=False)
df.head()

plot_and_save('Problem 2 QL time by states',
              data=df[df.solver == 'QLearning'],
              x='states', y='time', hue='actions')

# +
cols = ['states', 'actions']
test = df.groupby(cols)['policy'].agg(lambda x: np.equal(*x).mean())
test = test.to_frame().reset_index()

plot_and_save('Problem 2 QL policy agreement with VI',
              data=test,
              drawstyle='steps-pre',
              x='states', y='policy', hue='actions',
             ylabel='% agreement in policy function')
# -

P, R = mdptoolbox.example.rand(3, 4)
mod = mdptoolbox.mdp.QLearning(P, R, 0.95)
mod.run()

policy = mod.policy

policy

P[0, 1, :]

R[0, 0, :]

# +
D = df.copy()

cols = ['states', 'actions', 'solver']
D['value_function'] = D['value_function'].apply(lambda x: max(x))

D[cols + ['value_function']]


# +
def simulate(P, R, policy, discount=0.95, N_iter=1000, seed=None):
    """simulate the MDP with some policy"""
    N_states = len(policy)
    states_index = np.arange(N_states)
    
    if seed is not None:
        np.random.seed(seed)
    state = np.random.choice(states_index)
    
    value = 0
    for i in range(N_iter):
        action = policy[state]
        new_state = np.random.choice(states_index, p=P[action, state, :])
        reward = R[action, state, new_state]
        
        value += reward * (discount**i)
        state = new_state
        
    return value

# simulate(P, R, mod.policy, seed=0)


# +
states = 2048
actions = 16
discount = 0.95

np.random.seed(0)
P, R = mdptoolbox.example.rand(states, actions)

policies = {}
for solver in (mdptoolbox.mdp.QLearning, mdptoolbox.mdp.ValueIteration, mdptoolbox.mdp.PolicyIteration):
    
    params = {
        'discount': discount
    }
    if solver.__name__ == 'QLearning':
        params['n_iter'] = 100000
    
    mod = solver(P, R, **params)
    mod.run()
    
    policies[solver.__name__] = mod.policy
    
    del mod
    gc.collect()
    
    
# -

comp = []
for seed in range(1000):
    for k, v in policies.items():
        comp.append({
            'seed': seed,
            'solver': k,
            'score': simulate(P, R, policy=v, discount=discount, seed=seed)
        })

comp_df = pd.DataFrame(comp)

plot_and_save('Problem 2 QL vs VI Utility 1000 simulations',
              data=comp_df[comp_df.solver != 'PolicyIteration'],
              kind=sns.histplot,
              x='score', hue='solver', element='step', stat='percent',
             ylabel=None)




