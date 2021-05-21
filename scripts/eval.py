import os
import numpy as np
import pandas as pd 
import seaborn as sns
import sys
import nav as nv
import torch
from matplotlib import pyplot as plt
import matplotlib.colors


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = np.flip(rgb, axis=0)
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap


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

def gen_plot(sample_deviation, policies=None, save_path=None):
	base_dir = 'infer/' + str(sample_deviation) + '_deg'
	dfs = []
	if policies is None:
		policies = ['2k_ptz_lstm_1', '2k_ptz_lstm_5', '2k_ptz_lstm_15',
					'2k_lstm_1', '2k_lstm_5', '2k_lstm_15',
					'30k_lstm_1', '30k_lstm_5', '30k_lstm_15', 
					'60k_lstm_1', '60k_lstm_5', '60k_lstm_15', ]
	for env in ['Eastville', 'Hambleton', 'Hometown', 'Pettigrew', 'Beach']:
		for policy in policies:
			dfs.append(load_traj(base_dir, env, policy))
	df = pd.concat(dfs, ignore_index=True)
	cmap = categorical_cmap(4, 3, cmap='tab10')
	# sns.catplot(data=df, kind='bar', x='env', y='succ', hue='policy', 
				# palette=cmap(np.linspace(0, 1, cmap.N)))
	df['x'] = 'policy'
	sns.set(font_scale=1.)
	sns.set_style({'font.family':'serif', 'font.serif':['Times New Roman']})

	plot = sns.catplot(data=df, kind='bar', x='x', y='succ', hue='policy',
						palette=cmap(np.linspace(0, 1, cmap.N)))
	plot.set(xlabel=None)
	if save_path is not None:
		plot.savefig(os.path.join(save_path, 'inf_res.pdf'))



def infer(sample_deviation, policies=None): # policy argument expects a list
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

		nav = nv.init(env, 'infer/' + str(sample_deviation) + '_deg', policies=policies, sample_deviation=sample_deviation)
		for i in range(30):
			nav.compare_infer(env, i, sample_region[env], init_sep_steps=5, tags=policies)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--infer', action='store_true')
	parser.add_argument('--deg', type=int, default=360, help='sample deviation in degrees')
	parser.add_argument('--plot', action='store_true')
	parser.add_argument('--policy', nargs='+', default=None)
	parser.add_argument('--save_path', type=str, default=None)
	args = parser.parse_args()

	if args.infer:
		infer(args.deg, args.policy)

	if args.plot:
		gen_plot(args.deg, args.policy, args.save_path)
		plt.show()
