#!/usr/bin/env python
import os

serial_aims = '~/xing/apps/fhi-aims.160328/bin/aims.160328.serial.x'
parallel_aims = '/xing/apps/fhi-aims.160328/bin/aims.160328.mpi.x'

#exitcode = os.system(serial_aims)

parcmd = 'mpirun -np %i %s > aims.out' % (4,parallel_aims)
exitcode = os.system(parcmd)

#end
