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
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

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
        time.strftime("%m-%d-%Hh%Mm")
        )
    w : SummaryWriter = SummaryWriter(filename)
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(1, world.TRAIN_epochs + 1):
        start = time.time()
        if world.GPU:
            torch.cuda.reset_peak_memory_stats()
        output_information = Procedure.train(dataset, 
            sg_model, 
            loss_obj, 
            epoch, 
            writer=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        if epoch % 1 == 0:
            Procedure.test(dataset, sg_model, epoch, w)
        torch.save(sg_model.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()
