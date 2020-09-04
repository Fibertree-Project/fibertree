import subprocess
import sys

# given a frontier and graph, run all U_ formats on the configuration
frontier = sys.argv[1]
graph = sys.argv[2]
top_format = "U"
formats = ["U", "C", "H", "T", "B"]
for i in range(0, len(formats)):
    descriptor = "U" + formats[i]
    process = subprocess.Popen(['python3', 'codec-nknk.py', descriptor, frontier, graph, '>', 'out'])
