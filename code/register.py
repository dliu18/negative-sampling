import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ["Cora", "CiteSeer", "PubMed", "ego-facebook", "soc-ca-astroph"] or "SBM" in world.dataset:
	dataset = dataloader.SmallBenchmark(name=world.dataset, seed=world.seed)
elif world.dataset in ["ogbl-ppa", "ogbl-collab", "ogbl-citation2"]:
	dataset = dataloader.OGBBenchmark(name=world.dataset, seed=world.seed)
else:
	raise NotImplementedError(f"Haven't supported {world.dataset} yet!")

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print('===========end===================')