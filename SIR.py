import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import colors

df = pd.read_csv('Allmp.allfeatures.txt', sep='\t')


def negative_log_likelihood(params, mp_abundance, good_bacteria_abundance):
    beta, gamma = params
    n = len(mp_abundance)

    # S_prob = mp_abundance

    k = 1

    S_prob = k * mp_abundance

    infection_likelihood = binom.pmf(np.round(S_prob * n), n, beta)
    recovery_likelihood = binom.pmf(np.round((1 - S_prob) * n), n, gamma)

    log_likelihood = -np.sum(np.log(infection_likelihood + 1e-10)) - np.sum(np.log(recovery_likelihood + 1e-10))
    return log_likelihood


groups = df['Group'].unique()
r0_values = {}

for group in groups:
    group_data = df[df['Group'] == group]
    result = minimize(negative_log_likelihood, [0.1, 0.1],
                      args=(group_data['g__Mycoplasmoides'], group_data['GB']),
                      bounds=[(0, 1), (0, 1)])
    beta, gamma = result.x
    r0_values[group] = beta / gamma

    print(f"{group}: γ: {gamma}, β: {beta}, {r0_values[group]}")

    N = 400
    initial_infected = 20
    second_infected = 10
    time_steps = 25

    grid = np.zeros(N, dtype=int)
    infected_indices = np.random.choice(N, initial_infected, replace=False)
    grid[infected_indices] = 1

    cmap = colors.ListedColormap(['blue', 'red', 'green'])
    bounds = [0, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    history = [grid.copy()]
    infection_ratios = []

    for t in range(time_steps):
        new_grid = grid.copy()

        for i in range(N):
            if grid[i] == 0:
                if np.random.rand() < beta * np.sum(grid == 1) / N:
                    new_grid[i] = 1
            elif grid[i] == 1:
                if np.random.rand() < gamma:
                    new_grid[i] = 2

        if np.sum(new_grid == 1) == 0 and np.sum(new_grid == 0) >= second_infected:
            supplement_indices = np.random.choice(np.where(new_grid == 0)[0], second_infected, replace=False)
            new_grid[supplement_indices] = 1

        susceptible_and_infected = np.sum(new_grid == 0) + np.sum(new_grid == 1)
        infection_ratio = np.sum(new_grid == 1) / susceptible_and_infected if susceptible_and_infected > 0 else 0
        infection_ratios.append(infection_ratio)

        grid = new_grid.copy()
        history.append(grid.copy())

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle(f'{group} SIR in {time_steps} steps', fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        if i < len(history):
            reshaped_grid = history[i].reshape((20, 20))
            ax.imshow(reshaped_grid, cmap=cmap, norm=norm)
            ax.set_title(f'Time = {i}')
            ax.axis('off')

    plt.show()

    plt.figure(figsize=(5, 5))
    plt.plot(range(time_steps), infection_ratios, label='Infection Ratio')
    plt.xlabel('Time')
    plt.ylabel('Infection Ratio')
    plt.title(f'Infection Ratio over Time for {group}')
    plt.legend()
    plt.show()

df['R0'] = df['Group'].map(r0_values)

features = df.drop(columns=['Tag', 'g__Mycoplasmoides', 'GB', 'Group', 'R0'])
target = df['R0']

forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(features, target)
importances = forest.feature_importances_

importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

importance_df.to_csv('R0.features.tsv', sep='\t')

groups = list(r0_values.keys())

plt.figure(figsize=(10, 10))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title(f'Feature Importance for R0 ({groups[0]} vs {groups[1]})')
plt.gca().invert_yaxis()
plt.show()