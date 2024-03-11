import argparse
import birdepy as bd
import numpy as np
import os
import mlflow
import time
import matplotlib.pyplot as plt

start_time = time.time()
start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

# input and output arguments
parser = argparse.ArgumentParser()
parser.add_argument("--rate_of_spread", type=float, help="input: rate of spread")
parser.add_argument("--artefacs_folder", type=str, help="output: folder to save artefacts")

args = parser.parse_args()

print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

mlflow.start_run()
mlflow.set_experiment(experiment_name="eli-bird-model")


# Model parameters
model = "Verhulst"
rate_of_spread = args.rate_of_spread
recovery_rate = 0.25
population_size = 1000
true_parameters = [rate_of_spread, recovery_rate, 1 / population_size, 0]
simulation_horizon = 100
initial_number_infected = 10
obs_times = np.arange(0, simulation_horizon + 1, 1)

# Simulate the model
# print("<<< Simulating the model - start >>>")
number_infected = bd.simulate.discrete(
    true_parameters, model, initial_number_infected, obs_times, seed=2021
)
# print("<<< Simulating the model - end >>>")
total_number_infected = sum(number_infected)
mlflow.log_metric("total_number_infected", total_number_infected)
print(" ")
print(f"<<< Total number of infected: {total_number_infected} >>>")
print(" ")
# output paths are mounted as folder
np.savetxt(
    os.path.join(args.artefacs_folder, f"number_infected{rate_of_spread}.csv"),
    number_infected,
    delimiter=",",
)

# A plot
plt_filename = os.path.join(
    args.artefacs_folder, f"infected_population{rate_of_spread}.jpg"
)
plt.step(obs_times, number_infected, "r", where="post", color="tab:purple")
plt.title(f"rate_of_spread = {rate_of_spread}")
plt.ylabel("infected population")
plt.xlabel("day")
plt.savefig(plt_filename)

end_time = time.time()
end_time_str = time.strftime("%H:%M:%S", time.localtime(end_time))
elapsed_time = end_time - start_time
print(f"start_time: {start_time_str} -> end_time: {end_time_str}")
print(f"The program took {elapsed_time:.2f} seconds to run.")

mlflow.log_metric("elapsed_time", elapsed_time)
mlflow.end_run()
