import subprocess
import sys
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='manange parameters for training')
    parser.add_argument('--cfg',
                        help='the default setting is sdnet18 for cifar10 if no specific .yaml be chosen',
                        type=str,
                        )
    parser.add_argument('--yitaA',
                        help='the yitaA approx for each experiment',
                        type=int,
                        )

    parser.add_argument('--tau',
                        help='the tau for each experiment',
                        type=int,
                        )
    
    parser.add_argument('--initial_num',
                        help='the initial_num for powerIteration',
                        type=float,)
    args = parser.parse_args()

    return args

def generate_command(yita_a, tau, initial_num, num_layers, DATASET='cifar10', modelName='fwnet18', 
                        adjust_parm=None):
    dir_phase = DATASET+ '_'+ modelName
    modelLog = "logs"
    if adjust_parm != None:
        target_dir = DATASET+ "Logs/" +adjust_parm 
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        trainlog = target_dir +'/' +f"{num_layers}_yitaA{yita_a}W2_tau{tau}_initial{round(initial_num, 5)}"
    else:
        trainlog = DATASET+ "Logs/"+ f"{num_layers}_yitaA{yita_a}W2_tau{tau}_initial{round(initial_num, 5)}"
    command = f"python train.py --cfg ./experiments/{DATASET}.yaml --dir_phase {dir_phase} LOG_DIR ./{modelLog} \
        MODEL.NAME {modelName} MODEL.YITA_A {yita_a} MODEL.TAU {tau} MODEL.INITIAL_NUM {initial_num} MODEL.NUM_LAYERS {num_layers} \
            > ./{trainlog}.log 2>&1"
            
    return command

def main():
    # yitaA = list(range(2, 100, 3))
    yitaA = [1, 10, 100, 1000, 10000]
    tau = list(range(10, 1000, 10))
    initial_num = 1/8
    num_layers = 3
    for i in yitaA:
        command = generate_command(i, 100, initial_num, num_layers, DATASET='cifar100', adjust_parm='yitaA')
        run_command(command)
    
    for j in tau:
        command = generate_command(8, j, initial_num, num_layers, DATASET='cifar100', adjust_parm='tau')
        run_command(command)
    
def run_command(command):
    process = subprocess.run(
    command,
    shell=True,                  # 如果命令需要作为 shell 命令执行
    check=True,                  # 如果 return code 不为 0，抛出异常
)
    
    
    if process.returncode == 0:
        print(f"{command} completed successfully.")
    else:
        print(f"Process exited with return code {process.returncode}")

if __name__ == "__main__":
    main()