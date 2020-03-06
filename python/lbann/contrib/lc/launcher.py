from lbann.contrib.lc.systems import *
import lbann.launcher
from lbann.util import make_iterable

def run(*args, **kwargs):
    """Run LBANN with LC-specific optimizations (deprecated).

    This is deprecated. Use `lbann.contrib.launcher.run` instead.

    """

    import warnings
    warnings.warn(
        'Using deprecated function `lbann.contrib.lc.launcher.run`. '
        'Use `lbann.contrib.launcher.run` instead.'
    )
    from ..launcher import run as _run
    _run(*args, **kwargs)

def make_batch_script(
    system=system(),
    procs_per_node=procs_per_node(),
    scheduler=scheduler(),
    partition=partition(),
    account=account(),
    launcher_args=[],
    environment={},
    *args,
    **kwargs,
):
    """Construct batch script manager with LC-specific optimizations.

    This is a wrapper around `lbann.launcher.make_batch_script`, with
    defaults and optimizations for LC systems. See that function for a
    full list of options.

    """

    # Create shallow copies of input arguments
    launcher_args = list(make_iterable(launcher_args))
    environment = environment.copy()

    # Setup GPU bindings
    # Note: Each Hydrogen process is assigned to the GPU index that
    # matches its node communicator rank. This is not compatible with
    # mpibind, which assigns a GPU with index 0 to each process. We
    # can't use an exclusive GPU compute mode since processes may
    # touch the wrong GPU while figuring out ownership.
    if scheduler == 'slurm' and has_gpu(system):
        launcher_args.extend(['--mpibind=off',
                              '--nvidia_compute_mode=default'])

    # Optimized thread affinity for Pascal
    # Note: Both GPUs are on socket 0, so we only use cores on that
    # socket.
    if system == 'pascal':
        cores_per_socket = cores_per_node(system) // 2
        cores_per_proc = cores_per_socket // procs_per_node
        if 'AL_PROGRESS_RANKS_PER_NUMA_NODE' not in environment:
            environment['AL_PROGRESS_RANKS_PER_NUMA_NODE'] = procs_per_node
        if 'OMP_NUM_THREADS' not in environment:
            environment['OMP_NUM_THREADS'] = cores_per_proc - 1
        if scheduler == 'slurm':
            masks = [2**cores_per_proc - 1]
            while len(masks) < procs_per_node:
                masks.append(masks[-1] << cores_per_proc)
            mask_str = ','.join([hex(mask) for mask in masks])
            launcher_args.append('--cpu_bind=mask_cpu:{}'.format(mask_str))

    # Hacked bugfix for MPI_Init in MVAPICH2-2.3 (8/23/18)
    # Note: MPI_Init hangs when started with more than 35
    # processes. This bug is not present in MVAPICH2-2.2 but is
    # present in MVAPICH2-2.3rc2.
    if 'MV2_USE_RDMA_CM' not in environment:
        environment['MV2_USE_RDMA_CM'] = 0

    # Optimizations for Sierra-like systems
    if system in ('sierra', 'lassen'):

        # Set thread affinity
        # Note: Aluminum's default thread affinity is incorrect since
        # hwloc treats GPUs as NUMA domains.
        # Note: There are actually 22 cores/socket, but it seems that
        # powers of 2 are better for performance.
        cores_per_socket = 16
        procs_per_socket = (procs_per_node + 1) // 2
        cores_per_proc = cores_per_socket // procs_per_socket
        if 'AL_PROGRESS_RANKS_PER_NUMA_NODE' not in environment:
            environment['AL_PROGRESS_RANKS_PER_NUMA_NODE'] = procs_per_socket
        if 'OMP_NUM_THREADS' not in environment:
            environment['OMP_NUM_THREADS'] = cores_per_proc
        if scheduler == 'lsf':
            launcher_args.append('--bind packed:{}'.format(cores_per_proc))

        # Hack to enable process forking
        # Note: InfiniBand is known to experience hangs if an MPI
        # process is forked (see
        # https://www.open-mpi.org/faq/?category=openfabrics#ofa-fork).
        # Setting IBV_FORK_SAFE seems to fix this issue, but it may
        # hurt performance (see
        # https://linux.die.net/man/3/ibv_fork_init).
        if 'IBV_FORK_SAFE' not in environment:
            environment['IBV_FORK_SAFE'] = 1

        # Hacked bugfix for hcoll (1/23/19)
        # Note: Fixes hangs in MPI_Bcast.
        if 'HCOLL_ENABLE_SHARP' not in environment:
            environment['HCOLL_ENABLE_SHARP'] = 0
        if 'OMPI_MCA_coll_hcoll_enable' not in environment:
            environment['OMPI_MCA_coll_hcoll_enable'] = 0

        # Hacked bugfix for Spectrum MPI PAMI (9/17/19)
        if 'PAMI_MAX_NUM_CACHED_PAGES' not in environment:
            environment['PAMI_MAX_NUM_CACHED_PAGES'] = 0

        # Configure NVSHMEM to load Spectrum MPI
        if 'NVSHMEM_MPI_LIB_NAME' not in environment:
            environment['NVSHMEM_MPI_LIB_NAME'] = 'libmpi_ibm.so'

    return lbann.launcher.make_batch_script(
        procs_per_node=procs_per_node,
        scheduler=scheduler,
        partition=partition,
        account=account,
        launcher_args=launcher_args,
        environment=environment,
        *args,
        **kwargs,
    )
