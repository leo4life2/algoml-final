import matplotlib.pyplot as plt
import numpy as np
import json

# Load the data from the two JSON files
with open('/Users/kingston/Desktop/origbert/bert-results.json', 'r') as file:
    bert_results = json.load(file)

with open('/Users/kingston/Desktop/bertlsh/bertlsh-results.json', 'r') as file:
    bertlsh_results = json.load(file)

# print("bert", bert_results), print("bertlsh", bertlsh_results)
# assert()
# Inverting the axis for perplexity, eval loss, and train loss instead of negating their values
# This means that we will keep the original metric names

# Reverting to original metric names
original_metrics = ["Eval Accuracy", "Inverted Eval Loss", "Inverted Perplexity", "Inverted Train Loss", "Train Samples/Sec", "Train Steps/Sec"]

# Inverting the axis for the relevant metrics for BERT
inverted_bert_data = [
    bert_results["eval_accuracy"],
    1 / bert_results["eval_loss"],
    1 / bert_results["perplexity"],
    1 / bert_results["train_loss"],
    bert_results["train_samples_per_second"],
    bert_results["train_steps_per_second"]
]

# Inverting the axis for the relevant metrics for BERT-LSH
inverted_bertlsh_data = [
    bertlsh_results["eval_accuracy"],
    1 / bertlsh_results["eval_loss"],
    1 / bertlsh_results["perplexity"],
    1 / bertlsh_results["train_loss"],
    bertlsh_results["train_samples_per_second"],
    bertlsh_results["train_steps_per_second"]
]

# Normalizing the inverted data
max_values_inverted = [max(inverted_bert_data[i], inverted_bertlsh_data[i]) for i in range(len(original_metrics))]
normalized_bert_data_inverted = [inverted_bert_data[i] / max_values_inverted[i] for i in range(len(original_metrics))]
normalized_bertlsh_data_inverted = [inverted_bertlsh_data[i] / max_values_inverted[i] for i in range(len(original_metrics))]

# Completing the loop for the radar plot
normalized_bert_data_inverted += normalized_bert_data_inverted[:1]
normalized_bertlsh_data_inverted += normalized_bertlsh_data_inverted[:1]

# Recomputing angles for the original number of variables
angles_original = np.linspace(0, 2 * np.pi, len(original_metrics), endpoint=False).tolist()
angles_original += angles_original[:1]

# Plotting the inverted axis radar chart
fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))

# Rotate the plot by 90 degrees (π/2 radians)
ax.set_theta_offset(np.pi / 6)
ax.set_theta_direction(1)

ax.fill(angles_original, normalized_bertlsh_data_inverted, color='red', alpha=0.25)
ax.fill(angles_original, normalized_bert_data_inverted, color='blue', alpha=0.25)

# Labels for each point with increased font size
ax.set_yticklabels([], fontsize="large")  # Increase fontsize for Y-tick labels
ax.set_xticks(angles_original[:-1])
ax.set_xticklabels(original_metrics, fontsize=20)  # Increase fontsize for X-tick labels

# Legend
plt.legend(["BERT-LSH", "Baseline BERT"], loc="upper right", bbox_to_anchor=(1.18, 1.15), fontsize=15)
# plt.title("Comparison of Training Benchmarks: Baseline BERT vs BERT-LSH", pad=30)
# print()


plt.show()
fig.savefig("/Users/kingston/Desktop/training_radar.png", bbox_inches='tight')
# 