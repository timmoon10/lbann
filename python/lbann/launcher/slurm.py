"""Utility functions for Slurm."""

import os
import subprocess
from lbann.util import make_iterable
from .batch_script import BatchScript

def _time_string(minutes):
    """Time D-hh:mm:ss format."""
    minutes = max(minutes, 0)
    seconds = int(round((minutes % 1) * 60))
    hours, minutes = divmod(int(minutes), 60)
    days, hours = divmod(hours, 24)
    return f'{days}-{hours:02}:{minutes:02}:{seconds:02}'

class SlurmBatchScript(BatchScript):
    """Utility class to write Slurm batch scripts."""

    def __init__(self,
                 script_file=None,
                 work_dir=os.getcwd(),
                 nodes=1,
                 procs_per_node=1,
                 time_limit=None,
                 job_name=None,
                 partition=None,
                 account=None,
                 launcher='srun',
                 launcher_args=[],
                 interpreter='/bin/bash'):
        """Construct Slurm batch script manager.

        Args:
            script_file (str): Script file.
            work_dir (str, optional): Working directory
                (default: current working directory).
            nodes (int, optional): Number of compute nodes
                (default: 1).
            procs_per_node (int, optional): Parallel processes per
                compute node (default: 1).
            time_limit (int, optional): Job time limit, in minutes
                (default: none).
            job_name (str, optional): Job name (default: none).
            partition (str, optional): Scheduler partition
                (default: none).
            account (str, optional): Scheduler account
                (default: none).
            launcher (str, optional): Parallel command launcher
                (default: srun).
            launcher_args (`Iterable` of `str`, optional):
                Command-line arguments to srun.
            interpreter (str, optional): Script interpreter
                (default: /bin/bash).

        """
        super().__init__(script_file=script_file,
                         work_dir=work_dir,
                         interpreter=interpreter)
        self.nodes = nodes
        self.procs_per_node = procs_per_node
        self.time_limit = time_limit
        self.job_name = job_name
        self.partition = partition
        self.account = account
        self.launcher = launcher
        self.launcher_args = launcher_args

        # Configure header with Slurm job options
        self.add_header_line(f'#SBATCH --chdir={self.work_dir}')
        self.add_header_line(f'#SBATCH --output={self.out_log_file}')
        self.add_header_line(f'#SBATCH --error={self.err_log_file}')
        self.add_header_line(f'#SBATCH --nodes={self.nodes}')
        self.add_header_line(f'#SBATCH --ntasks={self.nodes * self.procs_per_node}')
        if self.time_limit is not None:
            self.add_header_line(f'#SBATCH --time={_time_string(self.time_limit)}')
        if self.job_name:
            self.add_header_line(f'#SBATCH --job-name={self.job_name}')
        if self.partition:
            self.add_header_line(f'#SBATCH --partition={self.partition}')
        if self.account:
            self.add_header_line(f'#SBATCH --account={self.account}')

    def add_parallel_command(self,
                             command,
                             work_dir=None,
                             nodes=None,
                             procs_per_node=None,
                             time_limit=None,
                             job_name=None,
                             partition=None,
                             account=None,
                             launcher=None,
                             launcher_args=None):
        """Add command to be executed in parallel.

        The command is launched with srun. Parallel processes are
        distributed evenly amongst the compute nodes.

        Args:
            command (`str` or `Iterable` of `str`s): Command to be
                executed in parallel.
            work_dir (str, optional): Working directory.
            nodes (int, optional): Number of compute nodes.
            procs_per_node (int, optional): Number of parallel
                processes per compute node.
            time_limit (int, optional): Job time limit, in minutes.
            job_name (str, optional): Job name.
            partition (str, optional): Scheduler partition.
            account (str, optional): Scheduler account.
            launcher (str, optional): srun executable.
            launcher_args (`Iterable` of `str`s, optional):
                Command-line arguments to srun.

        """

        # Use default values if needed
        if work_dir is None:
            work_dir = self.work_dir
        if nodes is None:
            nodes = self.nodes
        if procs_per_node is None:
            procs_per_node = self.procs_per_node
        if time_limit is None:
            time_limit = self.time_limit
        if job_name is None:
            job_name = self.job_name
        if partition is None:
            partition = self.partition
        if account is None:
            account = self.account
        if launcher is None:
            launcher = self.launcher
        if launcher_args is None:
            launcher_args = self.launcher_args

        # Construct srun invocation
        args = [launcher]
        args.extend(make_iterable(launcher_args))
        args.append(f'--chdir={work_dir}')
        args.append(f'--nodes={nodes}')
        args.append(f'--ntasks={nodes * procs_per_node}')
        args.append(f'--ntasks-per-node={procs_per_node}')
        if time_limit is not None:
            args.append(f'--time={_time_string(time_limit)}')
        if job_name:
            args.append(f'--job-name={job_name}')
        if partition:
            args.append(f'--partition={partition}')
        if account:
            args.append(f'--account={account}')
        args.extend(make_iterable(command))
        self.add_command(args)

    def submit(self, overwrite=False):
        """Submit batch job to Slurm with sbatch.

        The script file is written before being submitted.

        Args:
            overwrite (bool): Whether to overwrite script file if it
                already exists (default: false).

        Returns:
            int: Exit status from sbatch.

        """

        # Construct script file
        self.write(overwrite=overwrite)

        # Submit batch script and pipe output to log files
        run_proc = subprocess.Popen(['sbatch', self.script_file],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    cwd=self.work_dir)
        out_proc = subprocess.Popen(['tee', self.out_log_file],
                                    stdin=run_proc.stdout,
                                    cwd=self.work_dir)
        err_proc = subprocess.Popen(['tee', self.err_log_file],
                                    stdin=run_proc.stderr,
                                    cwd=self.work_dir)
        run_proc.stdout.close()
        run_proc.stderr.close()
        run_proc.wait()
        out_proc.wait()
        err_proc.wait()
        return run_proc.returncode
