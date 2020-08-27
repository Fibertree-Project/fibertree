import subprocess

top_rank = "U"
lower_ranks = ['U', 'C', 'B', 'H', 'T']
refs = ['codec-nknk-ref.py', 'codec-knkn-ref.py']

for ref in refs:
    for rank in lower_ranks:
        descriptor = top_rank + rank
        process = subprocess.Popen(['python3', ref, descriptor])
