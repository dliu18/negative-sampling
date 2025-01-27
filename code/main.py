import world
import utils
from model import SGModel
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join

from torch.profiler import profile, record_function, ProfilerActivity

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

train_mlp_epochs = [1, 5, 10, 15, 20]

sg_model = SGModel(world.config, dataset)
sg_model = sg_model.to(world.device)

loss_obj = utils.Loss(sg_model, world.config)
if world.config["loss_func"] == "sg":
	loss_obj = utils.SkipGramLoss(sg_model, world.config)
elif world.config["loss_func"] == "sg_aug":
	loss_obj = utils.SkipGramAugmentedLoss(sg_model, world.config)    

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
	try:
		sg_model.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
		world.cprint(f"loaded model weights from {weight_file}")
	except FileNotFoundError:
		print(f"{weight_file} not exists, start from beginning")

Neg_k = 1

# init tensorboard
if world.tensorboard:
	filename = join(world.BOARD_PATH,
		world.dataset,
		world.config["base_model"],
		world.config["loss_func"],
		str(world.config["n_negative"]),
		str(world.config["lr"]),
		str(world.config["lambda"]),
		str(world.config["n2v_p"]),
		str(world.config["n2v_q"]),
		time.strftime("%m-%d-%Hh%Mm")
		)
	w : SummaryWriter = SummaryWriter(filename)
else:
	w = None
	world.cprint("not enable tensorflowboard")

try:
	if not world.BYPASS_SKIPGRAM:
		completed_batches = 0
		for epoch in range(1, world.TRAIN_epochs + 1):
			start = time.time()
			if world.GPU:
				torch.cuda.reset_peak_memory_stats()
			# with torch.profiler.profile(
			# 	activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
			# 	profile_memory=True,
			# 	record_shapes=True
			# ) as prof:
			output_information, completed_batches = Procedure.train(dataset, 
				sg_model, 
				loss_obj, 
				epoch,
				completed_batches, 
				writer=w)
			print(f'EPOCH[{epoch}/{world.TRAIN_epochs}] {output_information}')

			if epoch in train_mlp_epochs and world.config["test_set"] == "valid":
				plot = (epoch == world.TRAIN_epochs)
				Procedure.train_edge_classifier(dataset, sg_model, loss_obj, writer=w, plot=plot)
				Procedure.test(dataset, sg_model, epoch, w, use_classifier=True)
				torch.save(sg_model.state_dict(), weight_file)
			# print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

	if world.config["test_set"] == "test" or world.TRAIN_epochs not in train_mlp_epochs:
		Procedure.train_edge_classifier(dataset, sg_model, loss_obj, writer=w, plot=True)
		Procedure.test(dataset, sg_model, epoch, w, use_classifier=True)
		torch.save(sg_model.state_dict(), weight_file)

	# Procedure.train_edge_classifier(dataset, sg_model, loss_obj, writer=w, plot=True)
	# torch.save(sg_model.state_dict(), weight_file)
finally:
	if world.tensorboard:
		w.close()
