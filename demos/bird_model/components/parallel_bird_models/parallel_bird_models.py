import argparse
import birdepy as bd
import numpy as np
import os
import mlflow
import time
import csv
import matplotlib.pyplot as plt


def init():
    global start_time
    global start_time_str
    global args

    parser = argparse.ArgumentParser(
        allow_abbrev=False, description="ParallelRunStep Agent"
    )
    parser.add_argument("--output_folder", type=str)

    args, _ = parser.parse_known_args()

    global output_folder
    output_folder = args.output_folder

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))


def run(mini_batch):
    print(f"<<< mini_batch = {mini_batch} >>>")

    resultList = []

    for config_file in mini_batch:
        print(f"<<< config_file = {config_file} >>>")
        filename_without_ext = os.path.splitext(os.path.basename(config_file))[0]
        print(f"<<< filename_without_ext = {filename_without_ext} >>>")
        with open(config_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                # print(f"line[{i}] = {line}")
                if i == 1:
                    rate_of_spread = float(line[0])

        print(f"<<< rate_of_spread = {rate_of_spread} >>>")

        # Model parameters
        model = "Verhulst"
        # rate_of_spread = args.rate_of_spread
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
        print(f"<<< Total number of infected: {total_number_infected} >>>")

        resultList.append([filename_without_ext, rate_of_spread, total_number_infected])

        # output paths are mounted as folder
        filename = filename_without_ext+"_number_infected.csv"
        filename = os.path.join(output_folder, filename)
        np.savetxt(
            filename,
            number_infected,
            delimiter=",",
        )

        filename = filename_without_ext + "_results.csv"
        filename = os.path.join(output_folder, filename)
        print(f"<<< filename = {filename} >>>")
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Scenario", "rate_of_spread", "total_number_infected"])
            for row in resultList:
                writer.writerow(row)

        # A plot
        plt_filename = filename_without_ext + "_infected_population.jpg"
        plt_filename = os.path.join(output_folder, plt_filename)
        plt.step(obs_times, number_infected, "r", where="post", color="tab:purple")
        plt.title(f"Scenario {filename_without_ext}: rate_of_spread = {rate_of_spread}")
        plt.ylabel("infected population")
        plt.xlabel("day")
        plt.savefig(plt_filename)

    print(f"<<< resultList = {resultList} >>>")
    filename = "results.csv"
    filename = os.path.join(output_folder, filename)
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scenario", "rate_of_spread", "total_number_infected"])
        for row in resultList:
            writer.writerow(row)

    end_time = time.time()
    end_time_str = time.strftime("%H:%M:%S", time.localtime(end_time))
    elapsed_time = end_time - start_time
    print(f"start_time: {start_time_str} -> end_time: {end_time_str}")
    print(f"The mini_batch took {elapsed_time:.2f} seconds to run.")

    return resultList
