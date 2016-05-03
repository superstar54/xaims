#!/usr/bin/env python
import os
from xaims.xaimsrc import XAIMSRC

serial_aims = XAIMSRC['aims.executable.serial']
parallel_aims = XAIMSRC['aims.executable.parallel']

if 'PBS_NODEFILE' in os.environ:
    # we are in the queue. determine if we should run serial or parallel
    NPROCS = len(open(os.environ['PBS_NODEFILE']).readlines())

    if NPROCS == 1:
        # no question. running in serial.
        exitcode = os.system(serial_aims)
    else:
        # We are running some kind of parallel job. This script only
        # supports MPI. It used to support multiprocessing, but it was
        # confusing, so I have taken it out for now.
        parcmd = 'mpirun -np %i %s' % (NPROCS, parallel_aims)
        exitcode = os.system(parcmd)
else:
    # probably running at cmd line, in serial.
    exitcode = os.system(serial_aims)

# end
