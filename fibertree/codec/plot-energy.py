import matplotlib.pyplot as plt
import yaml
import sys
import os
import numpy as np
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
indir = sys.argv[1]

alldata = dict()
# go through and read all files with 'cache_'
for filename in os.listdir(indir):
    if filename.startswith('cache_'):
        with open(os.path.join(indir, filename)) as f:
            data = yaml.load(f)
            desc = filename.split('_')[-1]
            alldata[desc] = data

# print(alldata)
ind = np.arange(len(alldata))
plts = list()
data_to_plot = dict()
x_labels = list()
legend_labels = list()
for key in alldata:
    data = alldata[key]
    x_labels.append(key)
    # print(key)
    # print(data)
    for name in data:
        val = data[name]
        if name.startswith("Amplify") or name.startswith("Reduce"): # add into Z_buffer
            continue
        else: 
            if name in data_to_plot:
                data_to_plot[name].append(val)
            else:
                data_to_plot[name] = [val]

    # add swoop stats in post
    A_buffer_key = "A_buffer_access"
    data_to_plot[A_buffer_key][-1] += data["Amplify_K0"]
    data_to_plot[A_buffer_key][-1] += data["Amplify_K1"]
    
    Z_buffer_key = "Z_buffer_access"
    data_to_plot[Z_buffer_key][-1] += data["Amplify_N0"]
    data_to_plot[Z_buffer_key][-1] += data["Amplify_N1_Upd"]
    data_to_plot[Z_buffer_key][-1] += data['Reduce_K0']

# print(data_to_plot)
legend_colors = list()
data_types = list()
plts = list()
num_vals = 0
for key in data_to_plot:
    print("normalizing {}".format(key))    
    val = data_to_plot[key]
    if 'DRAM' in key: # scale up 
        for i in range(0, len(val)):
            val[i] = val[i] * 100

color_map = {"A_DRAM_access":"b", "A_buffer_access":"lightskyblue", 'B_DRAM_access':'green','B_buffer_access':'lime', 'Z_DRAM_access':'red', 'Z_buffer_access':'salmon'}
stacking_order = ["B_buffer_access", "B_DRAM_access", "Z_buffer_access",
        "Z_DRAM_access", "A_buffer_access", "A_DRAM_access"]
cumulative = [0]*len(data_to_plot[stacking_order[0]])
for key in stacking_order:
# for i in range(0, len(stacking_order)):
    # key = stacking_order[i]
    # print("plotting {}".format(key))
    val = data_to_plot[key]
    print("{}: {}".format(key, val))
    c = color_map[key]
    p = plt.bar(ind, val,bottom=cumulative,color=c)
    cumulative = [sum(x) for x in zip(cumulative, val)]
    plts.append(p)
    
    # if key in color_map:
    #     legend_colors.append(colors[color_map[key]])
    # else:
    legend_colors.append(p[0])
    
    # rename legend
    temp = key
    parts = temp.split("_")
    label = ""
    inp_name = ""
    if parts[0] == "A":
        inp_name = "Fr"
    elif parts[0] == "B":
        inp_name = "Gr"
    elif parts[0] == "Z":
        inp_name = "Fr'"

    inp_name += " " + parts[1]
    
    data_types.append(inp_name)
assert len(ind) == len(x_labels)

# print(colors)
# print(legend_colors)
if len(sys.argv) > 2:
    plt.ylim(0, int(sys.argv[2]))
_, top_ylim = plt.ylim()
# label UH
for r in plts[-1]:
    h = r.get_height()
    if h > top_ylim / 2:
        print("h {} over limit".format(h))
        plt.text(r.get_x() + r.get_width() / 2., top_ylim *.8, "{:.2e}".format(h) , ha="center", va="center", color="white",fontsize=10, fontweight="bold")

plt.xticks(ind, x_labels)
plt.legend(data_types)
# print(indir)
exp_name = indir[:-1].split('/')[-1]
# print(exp_name)
plt.savefig('energy_' + exp_name + '.png')
# plt.savefig('out.png')
