"""Configuration dictionary for submitting jobs

mode = queue   # this defines whether jobs are immediately run or queued
user.name = xingwang
user.email = xing.wang@psi.ch

queue.command = qsub
queue.options = -joe
queue.walltime = 168:00:00
queue.nodes = 1
queue.ppn = 1
queue.mem = 2GB
queue.jobname = None

check for $HOME/.xaimsrc
then check for ./.xaimsrc

Note that the environment variables AIMS_SERIAL and AIMS_PARALLEL can
also be used to identify the aims executables used by runxaims.py.

"""
import os

# default settings
XAIMSRC = {'aims.executable.serial':
          '/opt/kitchingroup/aims-5.3.5/bin/aims-vtst-serial-beef',
          'aims.executable.parallel':
          '/opt/kitchingroup/aims-5.3.5/bin/aims-vtst-parallel-beef-aimssol',
          'mode': 'queue',  # other value is 'run'
          'queue.command': 'qsub',
          'queue.options': '-joe',
          'queue.walltime': '168:00:00',
          'queue.nodes': 1,
          'queue.ppn': 1,
          'queue.mem': '2GB',
          'queue.jobname': 'None',
          'multiprocessing.cores_per_process': 'None',
          'vdw_kernel.bindat': '/opt/kitchingroup/aims-5.3.5/vdw_kernel.bindat',
          'restart_unconverged': True
          }


def read_configuration(fname):
    """Reads xaimsrc configuration from fname."""
    f = open(fname)
    for line in f:
        line = line.strip()

        if line.startswith('#'):
            pass  # comment
        elif line == '':
            pass
        else:
            if '#' in line:
                # take the part before the first #
                line = line.split('#')[0]
            key, value = line.split('=')
            XAIMSRC[key.strip()] = value.strip()

# these are the possible paths to config files, in order of increasing
# priority
config_files = [os.path.join(os.environ['HOME'], '.xaimsrc'),
                '.xaimsrc']

for cf in config_files:
    if os.path.exists(cf):
        read_configuration(cf)
