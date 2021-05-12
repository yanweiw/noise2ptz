import os
import numpy as np
import pandas as pd 
import seaborn as sns
import sys
import nav as nv
import torch

def load_traj(base_dir, env_name, policy_name):
	# load traj by env and policy into a dataframe
	traj_dir = os.path.join(base_dir, env_name, policy_name)
	traj_acts = sorted([d for d in os.listdir(traj_dir) if 'action.txt' in d])
	steps = []
	succs = []
	idxes = []
	for i, f in enumerate(traj_acts):
		file_path = os.path.join(traj_dir, f)
		step = len(np.loadtxt(file_path, 'str'))
		assert step <= 50
		steps.append(step)
		succs.append(int(step < 50))
		idxes.append(i)

	data = {'idx': idxes, 'succ': succs, 'step': steps}
	df = pd.DataFrame.from_dict(data)
	df['env'] = env_name
	df['policy'] = policy_name
	return df

def gen_plot(sample_deviation):
	base_dir = 'infer/' + str(sample_deviation) + '_deg'
	dfs = []
	for env in ['Eastville', 'Hambleton', 'Hometown', 'Pettigrew', 'Beach']:
		for policy in ['2k_ptz_ff_1', '2k_ptz_lstm_1', '2k_lstm_15', '60k_lstm_15']:
			dfs.append(load_traj(base_dir, env, policy))
	df = pd.concat(dfs, ignore_index=True)
	sns.catplot(data=df, kind='bar', x='env', y='succ', hue='policy')



def infer(sample_deviation):
	sample_region = {'Eastville': (0, 2, 3, 7),
					 'Hambleton': (-8, -2, 0, 2.1), 
					 'Hometown': (-0.5, 2, -6, 0), 
					 'Pettigrew': (-8.5, -6.5, 2, 9), 
					 'Beach': (-2.5, 0.5, -0.5, 1.5)}

	for env in ['Eastville', 'Hambleton', 'Hometown', 'Pettigrew', 'Beach']:
		try:
			nav.sim.close()
		except:
			pass

		nav = nv.init(env, 'infer/' + str(sample_deviation) + '_deg', sample_deviation=sample_deviation)
		for i in range(30):
			nav.compare_infer(env, i, sample_region[env], init_sep_steps=5)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--infer', action='store_true')
	parser.add_argument('--deg', type=int, default=360, help='sample deviation in degrees')
	parser.add_argument('--plot', action='store_true')
	args = parser.parse_args()

	if args.infer:
		infer(args.deg)

	if args.plot:
		gen_plot(args.deg)