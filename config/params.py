import utils as U

models = ["simnet", "protonet_1", "protonet_1"]
features = ["bit", "byte"]
precision = [16, 32]
supports = [1, 5, 10, 50]
classes_seen = [0.4, 0.5, 0.8]

for a in U.data.dproduct(models, features, precision, supports):
    print(a)
