
from torchdrug import transforms
from torchdrug import datasets
from torchdrug import data
from torchdrug import layers
from torchdrug.layers import geometry
from torchdrug import models
from torchdrug import tasks
from torchdrug import core
import torch
import csv

truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])


class EnzymeCommissionToy(datasets.EnzymeCommission):
    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/data/EnzymeCommission.tar.gz"
    md5 = "728e0625d1eb513fa9b7626e4d3bcf4d"
    processed_file = "enzyme_commission_toy.pkl.gz"
    test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]

dataset = EnzymeCommissionToy("~/protein-datasets/", transform=transform, atom_feature=None, 
                            bond_feature=None)
train_set, valid_set, test_set = dataset.split()
print("Shape of function labels for a protein: ", dataset[0]["targets"].shape)
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))


gearnet = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], num_relation=7,
                         batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], 
                              num_relation=7, edge_input_dim=59, num_angle_bin=8,
                              batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

task = tasks.MultipleBinaryClassification(gearnet, graph_construction_model=graph_construction_model, num_mlp_layer=3,
                                          task=[_ for _ in range(len(dataset.tasks))], criterion="bce", metric=["auprc@micro", "f1_max"])



optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, batch_size=4)
solver.train(num_epoch=10)

metric, pred, target, output_rep = solver.evaluate("valid")
output_data = output_rep.tolist()
target_data = target.tolist()
pred_data = pred.tolist()

with open('output_rep_ec.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(output_data)

print('output_rep_ec.csv saved')

with open('target_ec.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(target_data)

print('target_ec.csv saved')

with open('pred_ec.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(pred_data)

print('pred_ec.csv saved')

print(len(output_rep))
print(len(valid_set))

pdb_id = []
for id in dataset.indices:
    pdb_id.append(dataset.pdb_files[id])

with open('pdb_id.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(pdb_id)
print('pdb_id.csv saved')
