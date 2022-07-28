import os


def generate_run_ssh(cfg, port_id):
    lines = []
    with open('classification_ModelNet40/run.sh', 'r') as f:
        lines = f.readlines()
    lines[31] = 'COMMAND="python -W ignore main.py {}\n"'.format(cfg)

    with open(os.path.join('{}.sh').format(port_id), 'w', newline='\n') as f:
        for line in lines:
            f.writelines(line)


with open('exps.txt', 'r') as f:
    exp_id = 3000
    for line in f.readlines():
        line = line.strip()
        line_segs = line.split('|')
        if len(line_segs) > 3:
            line = line_segs[3].strip()
            generate_run_ssh(line, exp_id)
            exp_id += 1
