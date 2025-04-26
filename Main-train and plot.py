# ------------------------------
# Main: training loop
# ------------------------------
import matplotlib.pyplot as plt
if __name__ == "__main__":

    csv_path = "training_data_2_11.csv"
    episode_length = 504 # Set one episode per day(24), week(168), two weeks(336), three weeks(504), four weeks(672) as needed
    env = ac_env.AirConditioningEnv(csv_path, episode_length=episode_length)
    env = NormalizeObservation(env)

    # Get the dimensions of the environment state and action
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    episodes = 2000  # Total number of training rounds
    batch_size = 32
    losses_per_episode = []  # Used to record the average loss per round
    cumulative_rewards = []     # Record cumulative reward per episode
    avg_THI_deviation_per_episode = []
    avg_power_consumption_per_episode = []
    peak_power_consumption_per_episode = []

    # Training Cycles
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_losses = []  # The loss of all replays in this round
        cumulative_reward = 0   # Initialize cumulative reward for the episode
        THI_deviations = []   # |THI - 23|
        instantaneous_powers = []
        prev_energy = 0

        for time in range(episode_length):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            cumulative_reward += reward

            # |THI - 23|
            current_THI = info["THI"]
            THI_deviation = abs(current_THI - 23)
            THI_deviations.append(THI_deviation)

            # Calculate instantaneous energy consumption
            current_energy = info["energy"]
            instantaneous_power = current_energy - prev_energy
            instantaneous_powers.append(instantaneous_power)
            prev_energy = current_energy

            if np.isnan(reward):
                print("NaN detected in reward!")
                print(f"reward: {reward}")
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break
            loss = agent.replay(batch_size)
            if loss is not None:
                episode_losses.append(loss)

        # Calculate the average THI deviation for this round
        avg_THI_deviation = np.mean(THI_deviations) if THI_deviations else 0
        avg_THI_deviation_per_episode.append(avg_THI_deviation)

        # Calculate the average instantaneous energy consumption and peak energy consumption of this round
        avg_power_consumption = np.mean(instantaneous_powers) if instantaneous_powers else 0
        peak_power_consumption = np.max(instantaneous_powers) if instantaneous_powers else 0
        avg_power_consumption_per_episode.append(avg_power_consumption)
        peak_power_consumption_per_episode.append(peak_power_consumption)

        # Calculate the average loss of the current round
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses_per_episode.append(avg_loss)
        cumulative_rewards.append(cumulative_reward)
        print(f"Episode:{e+1:>4}/{episodes:<4}, Avg Loss: {avg_loss:.4f}, Cumulative Reward: {cumulative_reward:>10.4f}, "
          f"Avg |THI-23|: {avg_THI_deviation:.4f}, Avg Power: {avg_power_consumption:.4f}, Peak Power: {peak_power_consumption:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), losses_per_episode, label='Average Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), avg_THI_deviation_per_episode, label='Average |THI-23| per Episode')
plt.xlabel('Episode')
plt.ylabel('Average |THI-23|')
plt.title('Average |THI-23| Curve')
plt.legend()
plt.show()

# Plot Cumulative Reward Curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), cumulative_rewards, label='Cumulative Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Curve')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), avg_power_consumption_per_episode, label='Average Power Consumption per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Power Consumption')
plt.title('Average Power Consumption Curve')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes+1), peak_power_consumption_per_episode, label='Peak Power Consumption per Episode')
plt.xlabel('Episode')
plt.ylabel('Peak Power Consumption')
plt.title('Peak Power Consumption Curve')
plt.legend()
plt.show()