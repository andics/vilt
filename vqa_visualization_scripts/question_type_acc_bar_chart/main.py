import matplotlib.pyplot as plt
import numpy as np
import re

file1 = "/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Programming/vilt/vqav2_eval_results/10d_and_baseline/variable_q_type_val_10d_FOR_PLOT_GEN.txt"
file2 = "/home/projects/bagon/andreyg/Projects/Variable_Resolution_VQA/Programming/vilt/vqav2_eval_results/10d_and_baseline/equiconst_q_type_val_10d_FOR_PLOT_GEN.txt"

def parse_file(path):
    with open(path, "r") as file:
        data = file.readlines()

    question_type_accuracy_dict = {}
    for line in data:
        if ":" in line:
            question_type, accuracy_rate = line.split(":")
            question_type_accuracy_dict[question_type.strip()] = float(accuracy_rate.strip())
    return question_type_accuracy_dict


def compute_differences(dict1, dict2):
    differences = {}
    for key in dict1:
        if key in dict2:
            differences[key] = dict1[key] - dict2[key]
    return differences


def generate_plots(file1, file2):
    dict1 = parse_file(file1)
    dict2 = parse_file(file2)
    differences = compute_differences(dict1, dict2)

    top_5_file1 = sorted([(k,dict1[k],dict2[k]) for k, v in differences.items() if v > 0],
                         key=lambda x: abs(x[1]-x[2]), reverse=True)[:5]

    top_5_file2 = sorted([(k,dict1[k],dict2[k]) for k, v in differences.items() if v < 0],
                         key=lambda x: abs(x[1]-x[2]), reverse=True)[:5]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,10))

    axis_limit = max(max(top_5_file1, key=lambda item:item[1])[1],
                     max(top_5_file1, key=lambda item:item[2])[2],
                     max(top_5_file2, key=lambda item:item[1])[1],
                     max(top_5_file2, key=lambda item:item[2])[2]) + 3

    width = 0.35  # this makes the bar thinner

    bars1 = ax1.bar([tup[0] for tup in top_5_file1], [tup[1] for tup in top_5_file1], width, color='#3b5998', alpha=1, label='Variable')
    bars2 = ax1.bar([tup[0] for tup in top_5_file1], [tup[2] for tup in top_5_file1], width, color='#61bff2', alpha=1, label='Uniform')
    ax1.set_ylim(0, axis_limit)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Top 5 question types: Variable > Uniform')
    ax1.legend()

    bars4 = ax2.bar([tup[0] for tup in top_5_file2], [tup[2] for tup in top_5_file2], width, color='#61bff2', alpha=1, label='Uniform')
    bars3 = ax2.bar([tup[0] for tup in top_5_file2], [tup[1] for tup in top_5_file2], width, color='#3b5998', alpha=1,
                    label='Variable')
    ax2.set_ylim(0, axis_limit)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Top 5 question types: Variable < Uniform')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.show()

if __name__ == "__main__":
    generate_plots(file1 = file1, file2 = file2)