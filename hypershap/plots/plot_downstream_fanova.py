import json
import matplotlib.pyplot as plt
import numpy as np

from hypershap.precompute.pre_compute_downstream_hpo import create_step_plot

data_dir = "res/hpo_runs/fanova/"
file_name_pattern = "wref_fanova_vs_hypershap_downstream_{did}.json"

out_dir = "res/plots/yahpogym/downstream_fanova/"
datasets_to_plot = [ 126025, 126026, 126029, 146212, 167104, 167161, 167168, 167181, 167190, 168868, 168910, 189865, 189866, 189905]

def create_step_plot(instance, full_res, full_std, hpi_res, hpi_std, opt, **kwargs):
    bo = False

    # plot the data
    full_res = np.array(full_res)
    full_std = np.array(full_std)
    hpi_res = np.array(hpi_res)
    hpi_std = np.array(hpi_std)
    opt = np.array([float(opt)] * 200)
    x = np.arange(1, 200 + 1)

    plt.step(x, hpi_res, label="HyperSHAP")
    plt.fill_between(x, hpi_res - hpi_std, hpi_res + hpi_std, alpha=0.2, step="pre")

    plt.step(x, full_res, label="fANOVA")
    plt.fill_between(x, full_res - full_std, full_res + full_std, alpha=0.2, step="pre")

    plt.step(x, opt, label="Optimum")

    title = str(instance)
    if bo:
        title += " (BO)"
    else:
        title += " (RS)"

    plt.title(title)
    plt.xlabel("Number of Evaluations")
    # plt.xscale("log")
    plt.ylabel("Accuracy")

    plt.legend()
    keywords_for_filename = [str(instance), "downstream", "hpo", str(1000)]

    if bo:
        keywords_for_filename.append("bo")
    else:
        keywords_for_filename.append("rs")

    plt.savefig(out_dir + str(instance) + ".png")
    #plt.show()
    plt.close()

for did in datasets_to_plot:
    path = data_dir + file_name_pattern.format(did=did)
    print(path)
    data = json.load(open(path))
    bo = data["bo"]
    create_step_plot(did, data["fn_res"], data["fn_std"], data["hs_res"], data["hs_std"], data["full_opt"])
