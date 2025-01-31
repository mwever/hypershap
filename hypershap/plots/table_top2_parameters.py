import json

did_list = [126025, 126026, 126029, 146212, 167104, 167161, 167168, 189865, 189866]
main_dir = "res/hpo_runs/fanova/"

params = dict()
for did in did_list:
    map1 = json.load(open(main_dir + str(did) + "_param_names.json"))
    map2 = json.load(open(main_dir + str(did) + "_sense_param_names.json"))
    params[did] = map1 | map2


print("\\begin{tabular}{l  ll  ll  ll} \\toprule")
print("\\textbf{Dataset} & \\multicolumn{2}{c}{\\textbf{fANOVA}} & \\multicolumn{2}{c}{\\textbf{Sensitivity}} & \\multicolumn{2}{c}{\\tool}\\\\ \\midrule")

line_pattern = "& {} & {} & {} & {} & {} & {} \\\\"
for did in did_list:
    param = params[did]
    line = line_pattern.format(param["fanova"][0], param["fanova"][1], param["sense"][0], param["sense"][1], param["hypershap"][0], param["hypershap"][1])
    line = line.replace("_", "\\_")
    print("\\textbf{"+str(did)+"}" + line )
print("\\bottomrule")
print("\\end{tabular}")
