import matplotlib.pyplot as plt  
import numpy as np 

def main():
	PROCESSES_TO_TEST=[1,2,4,8,16,32]
	reward_averages = [-0.5735963, -1.6567822, -1.6567822, -0.44465628, -0.03814222, -0.27687317]
	reward_std = [0.7628876, 0.0022180295, 0.0022180295, 0.58471584, 0.005083843, 0.34892747]
	training_steps_per_second = [s / t for s,t in zip(steps_per_experiment, training_times)]

	plt.figure()
	plt.subplot(1,2,1)
	plt.errorbar(PROCESSES_TO_TEST, reward_averages, yerr=reward_std, capsize=2, c='k', marker='o')
	plt.xlabel('Processes')
	plt.ylabel('Average return')
	plt.subplot(1,2,2)
	plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
	plt.xticks(range(len(PROCESSES_TO_TEST)),PROCESSES_TO_TEST)
	plt.xlabel('Processes')
	plt.ylabel('Training steps per second')

if __name__ == '__main__':
	main()