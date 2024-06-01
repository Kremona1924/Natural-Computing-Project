import glob
import json
import pandas as pd
import matplotlib.pyplot as plt


#path to run folder
run_folder = r"logs\experiment01\run01"

file_path = run_folder + "/performance_*"
match_path = glob.glob(file_path)
with open(match_path[0], 'r') as f:
    data = json.load(f)

keys = list(data.keys())
pd_data = {key: data[key] for key in keys[:-4]}

df = pd.DataFrame.from_dict(pd_data, orient='index', columns=[''])
print(df)

graph_data = {key: data[key] for key in keys[-4:]}

# Plot each data series
for key, values in graph_data.items():
    plt.plot(values, label=key)

# Add legend
plt.legend()

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Values Over Time')

# Show plot
plt.show()