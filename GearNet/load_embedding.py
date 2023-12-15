import torchdrug as td
import torch
from collections import OrderedDict
from torchdrug import layers, models, tasks, datasets, data, metrics, utils, core, transforms
from torchdrug.layers import geometry
import util
import sys
import csv


truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])

dataset = datasets.AlphaFoldDB(path = '/Users/harper.h/Documents/cs224w/final_project/yeast_alphafold', transform=transform, atom_feature=None, 
                            bond_feature=None)
lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

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

task = tasks.DistancePrediction(gearnet, graph_construction_model=graph_construction_model)



optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, dataset, dataset, dataset, optimizer, batch_size=4)
solver.train(num_epoch=10)

metric, output_rep = solver.evaluate("valid")
output_data = output_rep.tolist()


with open('output_rep_ec.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(output_data)

print('output_rep_ec.csv saved')

print(len(output_rep))
print(len(valid_set))

pdb_id = []
for id in dataset.indices:
    pdb_id.append(dataset.pdb_files[id])

with open('pdb_id.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(pdb_id)
print('pdb_id.csv saved')