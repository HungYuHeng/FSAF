# Copyright (c) 2021
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
import multiprocessing as mp
from datetime import datetime
from fsaf.policies.policies import NeuralAF
from fsaf.RL.DQN import DQN
import shutil

from fsaf.RL.plot_learning_curve_online import plot_learning_curve_online
from gym.envs.registration import register, registry
import torch
rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fsaf")
torch.cuda.set_device(0)
lengthScale = [0.1,0.2,0.3,0.1,0.2,0.3,0.3,0.5,0.6]
dim = [3]
# specifiy environment
kernels = ["RBF"]*3+["Matern32"]*3+["SM"]*3
kernel = "Matern32"
inner_loop_steps = 5
env_spec = {
    "env_id": "FSAF-GP-v0",
    "D": dim[0],
    "f_type": "GP",
    "f_opts": {
                "kernel": kernel,
               "lengthscale_low": 0.05,
               "lengthscale_high": 0.6,
               "noise_var_low": 0.1, # 這裡怪怪的
               "noise_var_high": 0.1, 
               "signal_var_low": 1.0, 
               "signal_var_high": 1.0, 
               "min_regret": 1e-20,
               "mix_kernel": True,
               "periods":[0.3,0.6],
               "kernel_list" : kernels,
               "inner_loop_steps" : inner_loop_steps}, # K
    "features": ["posterior_mean", "posterior_std", "incumbent", "timestep_perc"], # "incumbent", "timestep_perc" 
    "T_min": 30,
    "T_max": 100, # Line 7,8 長度100
    "n_init_samples": 0,
    "pass_X_to_pi": False,
    # will be set individually for each new function to the sampled hyperparameters
    "kernel": kernel,
    "kernel_lengthscale": None,
    "kernel_variance": None,
    "noise_variance": None,
    "use_prior_mean_function": False,
    "local_af_opt": False, # 會不會走出這200的點
    "cardinality_domain": 200,
    "reward_transformation": "neg_log10"  # true maximum not known
}

# specifyparameters
n_iterations = 1000
batch_size = 128
n_workers = 5
arch_spec = 4 * [200] # 4 200 200 200 200 1
num_particles = 5 # 五個NN，用不同Loss
dqn_spec = { # DQN.py 用到
    "batch_size": batch_size,
    "max_steps": n_iterations * batch_size,
    "lr": 1e-3, # Line 18
    "inner_lr":1e-2, # Line 10, 15
    "gamma": 0.98, # discount factor
    "buffer_size":1e3, # replay buffer
    "prior_alpha":0.3, # prioritized  replay buffer
    "prior_beta":0.6, # prioritized  replay buffer 
    "outer_w":0.01, # 沒用到
    "n_steps":3, # TD3
    "task_size":3, # 9 個抽三個
    "max_norm":40, # max gradient = 40
    "target_update_interval":5, # Target nn update frequency
    "n_workers": n_workers, 
    "env_id": env_spec["env_id"],
    "seed": 0,
    "env_seeds": list(range(n_workers)),
    "policy_options": {
        "activations": "relu",
        "arch_spec": arch_spec, 
        "use_value_network": True,
        "t_idx": -2, # "incumbent(目前最好點的y值)", "timestep_perc(第87步=0.87)"
        "T_idx": -1,
        "arch_spec_value": arch_spec,
    },
    "kernels" : kernels,
    "lengthScale" : lengthScale,
    "num_particles" : num_particles,
    "ML" : False, # Testing 前要做的更新
    "inner_loop_steps":inner_loop_steps,
    "using_chaser":True, # Line 18
    "demo_prob" : 1/128, # Page 15, Demo replay 上的機率，

}


# register environment
if env_spec["env_id"] in registry.env_specs:
    del registry.env_specs[env_spec["env_id"]]
register(
    id=env_spec["env_id"],
    entry_point="fsaf.environment.function_gym:FSAF",
    max_episode_steps=env_spec["T_max"] if "T_max" in env_spec else env_spec["T"],
    reward_threshold=None,
    kwargs=env_spec
)

# log data and weights go here, use this folder for evaluation afterwards
logpath = os.path.join(rootdir, "log", env_spec["env_id"], datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
shutil.copytree(rootdir+"/RL",logpath+"/RL")
shutil.copytree(rootdir+"/policies",logpath+"/policies")
shutil.copytree(rootdir+"/environment",logpath+"/environment")

# set up policy
policy_fn = lambda observation_space, action_space, deterministic: NeuralAF(observation_space=observation_space,
                                                                            action_space=action_space,
                                                                            deterministic=deterministic,
                                                                            options=dqn_spec["policy_options"])

# do training
print("Training on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
dqn = DQN(policy_fn=policy_fn, params=dqn_spec, logpath=logpath, save_interval=10)
# learning curve is plotted online in separate process
p = mp.Process(target=plot_learning_curve_online, kwargs={"logpath": logpath, "reload": True})
p.start()
dqn.train()
p.terminate()
plot_learning_curve_online(logpath=logpath, reload=False)

