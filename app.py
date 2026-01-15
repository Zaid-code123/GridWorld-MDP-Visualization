import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

ROWS, COLS = 5, 5
ACTIONS = ['U', 'D', 'L', 'R']
ACTION_MAP = {'U':(-1,0), 'D':(1,0), 'L':(0,-1), 'R':(0,1)}

GOAL = (4,4)
NEGATIVE = (3,3)
OBSTACLES = [(1,1), (2,1)]

STEP_REWARD = -0.1

st.title("Grid World MDP")

gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.9)

V = np.zeros((ROWS, COLS))
policy = np.random.choice(ACTIONS, size=(ROWS, COLS))

def is_valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and (r,c) not in OBSTACLES

def reward(state):
    if state == GOAL:
        return 10
    if state == NEGATIVE:
        return -10
    return STEP_REWARD

def value_iteration():
    global V
    new_V = np.copy(V)
    for r in range(ROWS):
        for c in range(COLS):
            if (r,c) in [GOAL, NEGATIVE] or (r,c) in OBSTACLES:
                continue
            values = []
            for a in ACTIONS:
                dr, dc = ACTION_MAP[a]
                nr, nc = r+dr, c+dc
                if not is_valid(nr,nc):
                    nr, nc = r, c
                values.append(reward((nr,nc)) + gamma * V[nr,nc])
            new_V[r,c] = max(values)
    V = new_V

def extract_policy():
    global policy
    for r in range(ROWS):
        for c in range(COLS):
            if (r,c) in [GOAL, NEGATIVE] or (r,c) in OBSTACLES:
                continue
            best_val = -1e9
            for a in ACTIONS:
                dr, dc = ACTION_MAP[a]
                nr, nc = r+dr, c+dc
                if not is_valid(nr,nc):
                    nr, nc = r, c
                val = reward((nr,nc)) + gamma * V[nr,nc]
                if val > best_val:
                    best_val = val
                    policy[r,c] = a

st.button("Step", on_click=lambda: [value_iteration(), extract_policy()])

fig, ax = plt.subplots()
ax.imshow(V, cmap='coolwarm')

for r in range(ROWS):
    for c in range(COLS):
        if (r,c) in OBSTACLES:
            ax.text(c, r, "X", ha='center', va='center')
        elif (r,c) == GOAL:
            ax.text(c, r, "G", ha='center', va='center')
        elif (r,c) == NEGATIVE:
            ax.text(c, r, "N", ha='center', va='center')
        else:
            ax.text(c, r, policy[r,c], ha='center', va='center')

ax.set_xticks([])
ax.set_yticks([])
st.pyplot(fig)
