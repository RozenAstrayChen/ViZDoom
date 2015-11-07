#!/usr/bin/python

import numpy as np
from qengine import QEngine
from games import ShootingDotGame
from evaluators import MLPEvaluator
import random 
from time import time
from time import sleep
import itertools as it
import lasagne

def actions_generator(game):
	n = game.get_action_format()
	actions = []
	for perm in it.product([False,True],repeat= n):
		actions.append(perm)
	return actions

def create_mlp_evaluator(state_format, actions_number, batch_size):
	mlp_args = {}
	mlp_args["state_format"] = state_format
	mlp_args["actions_number"] = actions_number
	mlp_args["batch_size"] = batch_size
	mlp_args["hidden_units"] = 100
	mlp_args["learning_rate"] = 0.01
	mlp_args["hidden_layers"] = 1
	mlp_args["hidden_nonlin"] = lasagne.nonlinearities.tanh
	return MLPEvaluator(**mlp_args)

game_args = {}
game_args['width'] = 11
game_args['height'] = 3
game_args['hit_reward'] = 1.0
game_args['max_moves'] = 50
#should be positive cause it's treatet as a penalty
game_args['miss_penalty'] = 0.0
#should be negative cause it's treatet as a reward
game_args['living_reward'] = -0.05
game_args['random_background'] = False
game_args['ammo'] = np.inf
game = ShootingDotGame(**game_args)

engine_args = {}
engine_args["history_length"] = 1
engine_args["bank_capacity"] = 100000
engine_args["evaluator"] = create_mlp_evaluator
engine_args["game"] = game
engine_args['start_epsilon'] = 1.0
engine_args['epsilon_decay_start_step'] = 500000
engine_args['epsilon_decay_steps'] = 5000000
engine_args['actions_generator'] = actions_generator
engine_args['update_frequency'] = 5
engine_args['batch_size'] = 30
engine_args['gamma'] = 0.7

engine = QEngine(**engine_args)

epochs = np.inf
training_episodes_per_epoch = 1000
test_episodes_per_epoch = 100
stop_mean = game.average_best_result()
overall_start = time()
print "Learning..."

epoch = 0
while epoch < epochs:
	engine.learning_mode = True
	rewards = []
	start = time()
	for episode in range(training_episodes_per_epoch):
		r = engine.run_episode()
		rewards.append(r)
		
	end = time()
	print "\nEpoch",epoch
	print "Train:"
	print engine.get_actions_stats(True)
	print "steps:",engine._steps, ", mean:", np.mean(rewards), "eps:",engine._epsilon
	print "t:",round(end-start,2)
	#learning off
	if test_episodes_per_epoch >0:
		engine.learning_mode = False
		rewards = []
		start = time()
		for episode in range(test_episodes_per_epoch):
			r = engine.run_episode()
			rewards.append(r)
			
		end = time()
		print "Test"
		print engine.get_actions_stats(clear = True,norm = False)
		m = np.mean(rewards)
		print "steps:",engine._steps, ", mean:", m
		if m > stop_mean:
			print stop_mean,"mean reached!"
			break
	epoch += 1
overall_end = time()

print "Elapsed time:",overall_end-overall_start