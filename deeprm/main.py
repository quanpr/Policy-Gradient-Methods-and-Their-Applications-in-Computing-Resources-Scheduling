import gym
import parameters
from DeepRMEnv import DeepRMEnv
import pdb

pa = parameters.Parameters()

pa.simu_len = 50  # 1000
pa.num_ex = 100 # 100
pa.num_nw = 10
# pa.num_seq_per_batch = 20
# pa.output_freq = 50
# pa.batch_size = 10

# pa.max_nw_size = 5
# pa.job_len = 5
pa.new_job_rate = 0.9

pa.episode_max_length = 2000  # 2000

pa.compute_dependent_parameters()

env = DeepRMEnv(pa, play=False, repre='image', test_mode=True, log_path="./logs")

# random policy
env.reset()
for _ in range(100000):
    # env.render()
    state, reward, done, _ = env.step(env.action_space.sample()) # take a random action
    pdb.set_trace()
env.close()

# # sjf policy
# env.reset()
# reward_list = []
# avg_reward = []
# for _ in range(100000):
#     env.render()
#     image = env.observe() # be adviced: set image[>0] as 1
#     import numpy as np
#     candidate_image = np.hstack((image[:,10:110], image[:,120:220]))
#     job_length = np.sum(candidate_image, axis=0)
#     job_length[job_length==0] = 256
#     sjf_action = (np.argmin(job_length) // 10) % 10
#     ob, reward, done, _ = env.step(sjf_action)  # take an action
#     reward_list.append(reward)
#     if done:
#         avg_reward.append(np.sum(reward_list))
#         print(np.sum(reward_list))
#         reward_list = []
#     # print(sjf_action)
#     # env.render()
# env.close()
# # print(np.mean(avg_reward))