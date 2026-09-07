"""
Reinforcement Learning Laboratory - Experiment 1
Exploring Reinforcement Learning Environments using Gymnasium

Objective:
1. Setup and verify Gymnasium environment setup.
2. Initialize the CartPole-v1 environment.
3. Explore Observation and Action Spaces.
4. Execute a Random Agent for one full episode.
5. Capture environment render and trajectory plots into screenshots/ directory.

Author: RL Student / Lab Implementation
"""

import os
import sys

# Task 1: Import required libraries with error handling
try:
    import gymnasium as gym
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print("\n[Error] Missing required dependencies.")
    print(f"Details: {e}")
    print("\nPlease install the required packages using:")
    print("    pip install -r requirements.txt")
    print("Or individually via:")
    print("    pip install gymnasium numpy matplotlib pygame")
    sys.exit(1)


def print_task_header(task_title: str) -> None:
    """Helper function to print formatted section headers."""
    print("\n" + "=" * 60)
    print(f" {task_title}")
    print("=" * 60)


def save_environment_screenshot(env: gym.Env, output_path: str = "screenshots/cartpole_render.png") -> None:
    """Renders and saves a screenshot frame of the environment."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Render RGB frame
        frame = env.render()
        if frame is not None:
            plt.figure(figsize=(6, 4))
            plt.imshow(frame)
            plt.title("CartPole-v1 Environment Visual Frame", fontsize=12, fontweight='bold')
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"[Info] Environment screenshot saved to '{output_path}'")
    except Exception as err:
        print(f"[Warning] Could not save environment screenshot: {err}")


def save_trajectory_plot(observations: list, output_path: str = "screenshots/episode_trajectory.png") -> None:
    """Plots and saves state observation trajectories over episode steps."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        obs_arr = np.array(observations)
        steps = np.arange(1, len(obs_arr) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, obs_arr[:, 0], label="Cart Position", color="#1f77b4", linewidth=2)
        plt.plot(steps, obs_arr[:, 1], label="Cart Velocity", color="#ff7f0e", linestyle="--")
        plt.plot(steps, obs_arr[:, 2], label="Pole Angle (rad)", color="#2ca02c", linewidth=2)
        plt.plot(steps, obs_arr[:, 3], label="Pole Angular Velocity", color="#d62728", linestyle="--")
        
        plt.axhline(0, color="gray", linestyle=":", alpha=0.7)
        plt.xlabel("Step Number", fontsize=11, fontweight='bold')
        plt.ylabel("Observation Values", fontsize=11, fontweight='bold')
        plt.title("CartPole-v1 State Observation Trajectory", fontsize=13, fontweight='bold')
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[Info] Trajectory plot screenshot saved to '{output_path}'")
    except Exception as err:
        print(f"[Warning] Could not save trajectory plot screenshot: {err}")


def main() -> None:
    """Main execution workflow for RL Lab Experiment 1."""
    
    # -------------------------------------------------------------------------
    # Task 1 – Environment Setup & Library Verification
    # -------------------------------------------------------------------------
    print_task_header("Task 1 - Environment Setup")
    
    # Print installed Gymnasium version as required
    print(f"Gymnasium Version : {gym.__version__}")

    # -------------------------------------------------------------------------
    # Task 2 – Create and Initialize Environment
    # -------------------------------------------------------------------------
    print_task_header("Task 2 - Create and Initialize Environment")
    
    env_name = "CartPole-v1"
    env = None

    try:
        # Attempting to create environment with rgb_array render mode to enable screenshots & rendering
        try:
            env = gym.make(env_name, render_mode="human")
        except Exception as render_err:
            print(f"[Warning] Failed to initialize with render_mode='human': {render_err}")
            print("[Info] Falling back to render_mode='rgb_array'...")
            env = gym.make(env_name, render_mode="rgb_array")

        # Reset environment to get initial state observation and info dictionary
        # reset() returns a tuple: (observation, info)
        observation, info = env.reset()

        print("Environment Created Successfully\n")
        print("Initial Observation:")
        print(observation)
        print("\nEnvironment Information:")
        print(info)

        # ---------------------------------------------------------------------
        # Task 3 – Observation and Action Space Exploration
        # ---------------------------------------------------------------------
        print_task_header("Task 3 - Observation and Action Space Exploration")

        # Observation Space represents all possible states the agent can observe.
        # For CartPole-v1, this is a Box space containing 4 continuous variables:
        # [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity].
        print("Observation Space:")
        print(env.observation_space)

        # Action Space represents all discrete actions available to the agent.
        # Discrete(2) means 2 possible discrete actions:
        # 0 -> Push cart to the left
        # 1 -> Push cart to the right
        print("\nAction Space:")
        print(env.action_space)

        # Print the explicit Python class type of the observation space
        print("\nObservation Space Type:")
        print(type(env.observation_space))

        # Print the total count of discrete actions supported
        print("\nNumber of Possible Actions:")
        print(env.action_space.n)

        # ---------------------------------------------------------------------
        # Task 4 – Random Agent Execution
        # ---------------------------------------------------------------------
        print_task_header("Task 4 - Random Agent Execution")

        step_count = 0
        total_reward = 0.0
        terminated = False
        truncated = False

        obs_history = [observation]

        # Run one complete episode until either terminated or truncated
        while not (terminated or truncated):
            step_count += 1

            # Sample a random action from the action space
            action = env.action_space.sample()

            # Execute action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            obs_history.append(obs)

            # Format and display step details as specified in prompt
            print("---------------------------------")
            print(f"Step : {step_count}")
            print(f"Action : {action}")
            print("Observation :")
            print(obs)
            print(f"\nReward : {reward}")
            print(f"\nTerminated : {terminated}")
            print(f"\nTruncated : {truncated}")
            print("---------------------------------")

        # Display summary once the episode finishes
        print("\nEpisode Finished\n")
        print(f"Total Steps :\n{step_count}\n")
        print(f"Total Reward :\n{total_reward}")

        # Capture and save screenshots of environment state & step trajectory
        print_task_header("Saving Screenshots")
        
        # If env was human, switch to rgb_array env temporarily for screenshot rendering
        screenshot_env = gym.make(env_name, render_mode="rgb_array")
        screenshot_env.reset()
        save_environment_screenshot(screenshot_env, "screenshots/cartpole_render.png")
        screenshot_env.close()

        save_trajectory_plot(obs_history, "screenshots/episode_trajectory.png")

    except gym.error.Error as e:
        print(f"\n[Error] Gymnasium environment creation failed: {e}")
    except KeyboardInterrupt:
        print("\n[Info] Execution interrupted by user (KeyboardInterrupt). Exiting safely...")
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")
    finally:
        # Task 5 Additional Requirement: Always close environment safely in finally block
        if env is not None:
            env.close()
            print("\n[Info] Environment successfully closed.")


if __name__ == "__main__":
    main()
