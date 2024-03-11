'''
Ilustrate configuring scenarios where each scenario is a CSV file which will be picked up by the aml parallel run.
note: for a real solution I would use a yaml file to store the scenario parameters.
'''
import os
import time
import csv
import argparse

start_time = time.time()
start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_folder", type=str, help="output: folder to save artefacts"
)

args = parser.parse_args()

scenario_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.80, 0.90]
# scenario_list = [0.1, 0.75, 0.90]

# Generate a CSV file for each scenario
for i, value in enumerate(scenario_list, start=1):
    scenario_folder = f"scenario_{i:03d}"
    filename = f"scenario_{i:03d}.csv"
   
    # Putting in folders makes a difference - that is how the partitioning is done
    deep_folder = os.path.join(args.output_folder, scenario_folder)
    os.makedirs(deep_folder, exist_ok=True)
    
    filename_deep = os.path.join(args.output_folder, scenario_folder, filename)

    with open(filename_deep, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["rate_of_spread"])
        writer.writerow([value])

end_time = time.time()
end_time_str = time.strftime("%H:%M:%S", time.localtime(end_time))
elapsed_time = end_time - start_time
print(f"start_time: {start_time_str} -> end_time: {end_time_str}")
print(f"The program took {elapsed_time:.2f} seconds to run.")
