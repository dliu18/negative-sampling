import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ["Cora", "CiteSeer", "PubMed", "ego-facebook", "soc-ca-astroph"] or "SBM" in world.dataset:
	dataset = dataloader.SmallBenchmark(name=world.dataset, 
		test_set=world.config["test_set"],
		test_set_frac=world.config["test_set_frac"], 
		seed=world.seed)
elif world.dataset in ["ogbl-ppa", "ogbl-collab", "ogbl-citation2", "ogbl-vessel"]:
	dataset = dataloader.OGBBenchmark(name=world.dataset, test_set=world.config["test_set"], seed=world.seed)
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