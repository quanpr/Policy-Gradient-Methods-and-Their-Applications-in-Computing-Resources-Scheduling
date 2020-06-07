import numpy as np
import os

# log_path = "/home/ziqi/Desktop/deeprm-env/logs/"
# file_name = "test_job_record_rate_{}.npy".format(0.3)
log_path = "/home/prquan/Github/PPO-PyTorch/logs/"
file_name = "test_job_record_rate_{}.npy".format(0.4)

log_filename = os.path.join(log_path, file_name)
saved_log = np.load(log_filename, allow_pickle=True)

avg_slowdown = []
for exp_cnt in range(len(saved_log)):
    info = saved_log[exp_cnt]

    all_discount_rews = []
    jobs_slow_down = []
    work_complete = []
    work_remain = []
    job_len_remain = []
    num_job_remain = []

    enter_time = np.array([info.record[i].enter_time for i in range(len(info.record))])
    finish_time = np.array([info.record[i].finish_time for i in range(len(info.record))])
    job_len = np.array([info.record[i].len for i in range(len(info.record))])
    job_total_size = np.array([np.sum(info.record[i].res_vec) for i in range(len(info.record))])

    finished_idx = (finish_time >= 0)
    unfinished_idx = (finish_time < 0)

    jobs_slow_down.append(
        (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]
    )
    work_complete.append(
        np.sum(job_len[finished_idx] * job_total_size[finished_idx])
    )
    work_remain.append(
        np.sum(job_len[unfinished_idx] * job_total_size[unfinished_idx])
    )
    job_len_remain.append(
        np.sum(job_len[unfinished_idx])
    )
    num_job_remain.append(
        len(job_len[unfinished_idx])
    )
    avg_slowdown.append(np.mean(jobs_slow_down))
print(np.mean(avg_slowdown))


