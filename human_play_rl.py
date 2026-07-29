import logging
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Union
import pygame
from envs.SGAI_2048 import SGAI_2048, Act2048
from models.NTupleInference import NTupleInference
from utils.logger import logger
from utils.renderer import Board2048Renderer

logger.setLevel(logging.INFO)

# rl-headless keeps the env from drawing its own window, so the original 2048
# palette in utils/renderer.py is the only thing on screen
env = SGAI_2048(render_mode="rl-headless", num_rows=4, num_cols=4)
renderer = Board2048Renderer()

# Create Agents to play
# Only one game with the GUI, so search is affordable here. A budget of 2000
# costs about 20ms per move, well under the 4 fps the renderer runs at.
model_weights_path = os.path.join("models", "ntuple_files", "ntuple_final.npz")
rl_agent = NTupleInference(weights_file_path=model_weights_path, node_budget=2_000)
# model_weights_path = os.path.join('models', 'q_table_files', 'q_table_final.pickle')
# rl_agent = QAgentInference(q_table_file_path=model_weights_path)
# model_weights_path = os.path.join('models', 'dq_model_files', 'dq_policy_final.pth')
# rl_agent = DQAgentInference(model_file_path=model_weights_path)

# Things to keep track of, though there will only be one game at a time
turn_rewards: List[int] = []
overall_score: Union[int, None] = None
all_actions: Dict[Act2048, int] = {
    Act2048.UP: 0,
    Act2048.DOWN: 0,
    Act2048.LEFT: 0,
    Act2048.RIGHT: 0,
}

# Initialize env and counters for human game
obs_, info_ = env.reset()
run_: bool = True
turn_: int = 0
renderer.draw(env.board, env.total_score, status="press space to advance a turn",
              highest_tile=env.highest_tile)

# Start running the game until the game is done
while run_:

    action: Union[None, Act2048] = None
    show_next: bool = False
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            overall_score = env.total_score
            run_ = False
            break
        if event.type == pygame.KEYDOWN:
            if event.key in [
                pygame.K_KP_ENTER,
                pygame.K_SPACE,
                pygame.K_w,
                pygame.K_a,
                pygame.K_s,
                pygame.K_d,
                pygame.K_UP,
                pygame.K_DOWN,
                pygame.K_RIGHT,
                pygame.K_LEFT,
            ]:
                show_next = True
            if event.key in [pygame.K_q, pygame.K_ESCAPE, pygame.K_BACKSPACE]:
                action = None
                overall_score = env.total_score
                run_ = False
                break

    if not show_next:
        continue

    # Get action from the algo agent
    action: int = rl_agent.get_action(obs_raw=obs_)

    all_actions[Act2048(action)] += 1
    obs_, reward, terminated, truncated, info_ = env.step(action)
    turn_rewards.append(reward)
    renderer.draw(
        env.board,
        env.total_score,
        status=f"turn {turn_}   {Act2048(action).name.lower()}   {reward:+}",
        highest_tile=env.highest_tile,
    )
    print(
        f"Turn: {turn_}  -  step reward: {reward}  -  total reward: {env.total_score}"
    )

    if terminated or truncated:
        run_ = False
        overall_score = env.total_score
        renderer.draw_game_over(env.total_score)
        pygame.time.wait(1500)

    turn_ += 1

env.close()

# Plotting
if overall_score is not None:
    plt.bar(["Total Score"], [overall_score], color="blue")
    plt.ylabel("Score (merge points)")
    plt.title("Final Score")
    plt.show()

plt.plot(turn_rewards)
plt.xlabel("Turn")
plt.ylabel("Step Reward")
plt.title("Reward per Turn")
plt.show()

plt.bar(range(len(all_actions)), list(all_actions.values()), align="center")
plt.xticks(range(len(all_actions)), [str(act_) for act_ in list(all_actions.keys())])
plt.xlabel("Action")
plt.ylabel("Times Chosen")
plt.title("Action Counts")
plt.show()

print(f"Score: {overall_score}  -  Highest tile: {env.highest_tile}")
