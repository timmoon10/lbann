"""Learn embedding weights with LBANN."""
import argparse
import os.path

import lbann
import lbann.contrib.launcher
import lbann.contrib.args

import data.data_readers
import utils
import utils.snap

# ----------------------------------
# Options
# ----------------------------------

# Command-line arguments
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_node2vec', type=str,
    help='job name', metavar='NAME')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=1, type=int,
    help='number of epochs (default: 1)', metavar='NUM')
parser.add_argument(
    '--latent-dim', action='store', default=128, type=int,
    help='latent space dimensions (default: 128)', metavar='NUM')
parser.add_argument(
    '--learning-rate', action='store', default=-1, type=float,
    help='learning rate (default: 0.025*mbsize)', metavar='VAL')
parser.add_argument(
    '--work-dir', action='store', default=None, type=str,
    help='working directory', metavar='DIR')
parser.add_argument(
    '--offline-walks', action='store_true',
    help='perform random walks offline')
args = parser.parse_args()

# Default learning rate
# Note: Learning rate in original word2vec is 0.025
if args.learning_rate < 0:
    args.learning_rate = 0.025 * args.mini_batch_size

# ----------------------------------
# Create data reader
# ----------------------------------

# Properties for graph and random walk
# Note: Use parameters from offline walk script
import data.offline_walks
graph_file = data.offline_walks.graph_file
num_graph_nodes = data.offline_walks.max_graph_node_id() + 1
walk_length = data.offline_walks.walk_context_length
return_param = data.offline_walks.return_param
inout_param = data.offline_walks.inout_param
num_negative_samples = data.offline_walks.num_negative_samples

# Construct data reader
if args.offline_walks:
    reader = data.data_readers.make_offline_data_reader()
else:
    # Note: Before starting LBANN, we preprocess graph by ingesting
    # with HavoqGT and writing distributed graph to shared memory.
    distributed_graph_file = '/dev/shm/graph'
    reader = data.data_readers.make_online_data_reader(
        graph_file=distributed_graph_file,
        walk_length=walk_length,
        return_param=return_param,
        inout_param=inout_param,
        num_negative_samples=num_negative_samples,
    )

# ----------------------------------
# Embedding weights
# ----------------------------------

encoder_embeddings_weights = lbann.Weights(
    initializer=lbann.NormalInitializer(
        mean=0, standard_deviation=1/args.latent_dim,
    ),
    name='embeddings',
)
decoder_embeddings_weights = lbann.Weights(
    initializer=lbann.ConstantInitializer(value=0),
    name='decoder_embeddings',
)

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Embedding vectors, including negative sampling
# Note: Input is sequence of graph node IDs
input_ = lbann.Identity(lbann.Input())
input_slice = lbann.Slice(
    input_,
    slice_points=f'0 {num_negative_samples+1} {num_negative_samples+walk_length}'
)
decoder_embeddings = lbann.DistEmbedding(
    input_slice,
    weights=decoder_embeddings_weights,
    num_embeddings=num_graph_nodes,
    embedding_dim=args.latent_dim,
    sparse_sgd=True,
    learning_rate=args.learning_rate,
)
encoder_embeddings = lbann.DistEmbedding(
    input_slice,
    weights=encoder_embeddings_weights,
    num_embeddings=num_graph_nodes,
    embedding_dim=args.latent_dim,
    sparse_sgd=True,
    learning_rate=args.learning_rate,
)

# Skip-Gram with negative sampling
preds = lbann.MatMul(decoder_embeddings, encoder_embeddings, transpose_b=True)
preds_slice = lbann.Slice(
    preds,
    axis=0,
    slice_points=f'0 {num_negative_samples} {num_negative_samples+1}')
preds_negative = lbann.Identity(preds_slice)
preds_positive = lbann.Identity(preds_slice)
obj_positive = lbann.LogSigmoid(preds_positive)
obj_positive = lbann.Reduction(obj_positive, mode='sum')
obj_negative = lbann.WeightedSum(preds_negative, scaling_factors='-1')
obj_negative = lbann.LogSigmoid(obj_negative)
obj_negative = lbann.Reduction(obj_negative, mode='sum')
obj = [
    lbann.LayerTerm(obj_positive, scale=-1),
    lbann.LayerTerm(obj_negative, scale=-1/num_negative_samples),
]

# ----------------------------------
# Run LBANN
# ----------------------------------

# Create optimizer
opt = lbann.SGD(learn_rate=args.learning_rate)

# Create LBANN objects
trainer = lbann.Trainer(num_parallel_readers=0)
callbacks = [
    lbann.CallbackPrint(),
    lbann.CallbackTimer(),
    lbann.CallbackDumpWeights(basename='embeddings',
                              epoch_interval=args.num_epochs),
    lbann.CallbackPrintModelDescription(),
]
model = lbann.Model(args.mini_batch_size,
                    args.num_epochs,
                    layers=lbann.traverse_layer_graph(input_),
                    objective_function=obj,
                    callbacks=callbacks)

# Create batch script
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
script = lbann.contrib.launcher.make_batch_script(
    job_name=args.job_name,
    work_dir=args.work_dir,
    **kwargs,
)

# Preprocess graph data with HavoqGT if needed
if not args.offline_walks:
    ingest_graph_exe = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'build',
        'havoqgt',
        'src',
        'ingest_edge_list',
    )
    script.add_parallel_command([
        ingest_graph_exe,
        f'-o {distributed_graph_file}',
        f'-d {2**30}',
        '-u 1',
        graph_file,
    ])

# LBANN invocation
prototext_file = os.path.join(script.work_dir, 'experiment.prototext')
lbann.proto.save_prototext(
    prototext_file,
    trainer=trainer,
    model=model,
    data_reader=reader,
    optimizer=opt,
)
script.add_parallel_command([
    lbann.lbann_exe(),
    f'--prototext={prototext_file}',
    f'--num_io_threads=1',
])

# Run LBANN
script.run()
