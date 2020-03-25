"""Default settings for LC systems."""
import socket
import re

# ==============================================
# Set system parameters
# ==============================================

class SystemParams:
    """Simple data structure to describe an LC system."""
    def __init__(self, cores_per_node, gpus_per_node, scheduler):
        self.cores_per_node = cores_per_node
        self.gpus_per_node = gpus_per_node
        self.scheduler = scheduler

# Supported LC systems
_system_params = {
    'catalyst': SystemParams(24, 0, 'slurm'),
    'corona':   SystemParams(24, 0, 'slurm'),
    'pascal':   SystemParams(36, 2, 'slurm'),
    'quartz':   SystemParams(36, 0, 'slurm'),
    'surface':  SystemParams(16, 2, 'slurm'),
    'lassen':   SystemParams(44, 4, 'lsf'),
    'ray':      SystemParams(40, 4, 'lsf'),
    'sierra':   SystemParams(44, 4, 'lsf'),
}

# Detect system
_system = re.sub(r'\d+', '', socket.gethostname())
if _system not in _system_params.keys():
    _system = None

# ==============================================
# Access functions
# ==============================================

def system():
    """Name of LC system."""
    if _system:
        return _system
    else:
        raise RuntimeError('unknown system '
                           '(' + socket.gethostname() + ')')

def is_lc_system(system = system()):
    """Whether current system is a supported LC system."""
    return (system is not None) and (system in _system_params.keys())

def gpus_per_node(system = system()):
    """Number of GPUs per node."""
    if not is_lc_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].gpus_per_node

def has_gpu(system = system()):
    """Whether LC system has GPUs."""
    return gpus_per_node(system) > 0

def cores_per_node(system = system()):
    """Number of CPU cores per node."""
    if not is_lc_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].cores_per_node

def scheduler(system = system()):
    """Job scheduler for LC system."""
    if not is_lc_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].scheduler

def procs_per_node(system = system()):
    """Default number of processes per node."""
    if has_gpu(system):
        return gpus_per_node(system)
    else:
        # Catalyst and Quartz have 2 sockets per node
        ### @todo Think of a smarter heuristic
        return 2
