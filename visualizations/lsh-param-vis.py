import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# Function to read the data from a file and parse it into a pandas DataFrame
def read_and_parse_file(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data_list.append(ast.literal_eval(line))
    return pd.DataFrame(data_list)

# Reading and parsing the file
df = read_and_parse_file('/Users/kingston/Desktop/lsh-params.txt')

# Combine 'bands' and 'num_hashes' into a single string parameter for plotting
df['bands_num_hashes'] = df[['bands', 'num_hashes']].astype(str).agg('-'.join, axis=1)


# Plotting the bar chart
plt.figure(figsize=(8, 6))
barplot = sns.barplot(data=df, x='bands_num_hashes', y='KFLOPs', color="red", alpha=0.7)
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)

# Drawing a dotted line at y=25600
plt.axhline(y=25.600, color='b', linestyle='--', label='Full Self Attention')

# Annotating the line
plt.text(x=0, y=25.600, s=' Full Self Attention', color='blue', ha='right', backgroundcolor='white', fontsize=13)


# plt.title('KFLOPs for Combined Parameters (Bands-Num_Hashes)')
plt.xlabel('Combined Parameters (bands-numHashes)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('KFLOPs', fontsize=15)
plt.tight_layout()  # Adjust layout to fit the x-axis labels
plt.savefig("/Users/Kingston/Desktop/kflops.png")
plt.show()