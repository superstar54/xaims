from xaims import *
import uuid
import textwrap
import logging
log = logging.getLogger('Xaims')
# * Archive and clone
# http://cms.mpi.univie.ac.at/aims/aims/Files_used_aims.html
aimsfiles = ['control.in', 'geometry.in', 'geometry.in.next_step',
             'aims.out', 'parameters.ase'
             ]


def clone(self, newdir, extra_files=None):
    '''copy a aims directory to a new directory. Does not overwrite
    existing files. newdir is relative to the the directory the
    calculator was created from, not the current working directory,
    unless an absolute path is used.

    what to do about METADATA, the uuid will be wrong!
    '''
    if extra_files is None:
        extra_files = []

    if os.path.isabs(newdir):
        newdirpath = newdir
    else:
        newdirpath = os.path.join(self.cwd, newdir)

    import shutil
    if not os.path.isdir(newdirpath):
        os.makedirs(newdirpath)
    for vf in aimsfiles+extra_files:

        if (not os.path.exists(os.path.join(newdirpath, vf))
            and os.path.exists(vf)):
            shutil.copy(vf, newdirpath)


    os.chdir(self.aimsdir)

Aims.clone = clone



def calculate(self, atoms=None, properties=['energy'],
                  system_changes=None):
    '''
    monkeypatched function to avoid calling calculate unless we really
    want to run a job. If a job is queued or running, we should exit
    here to avoid reinitializing the input files.

    I also made it possible to not give an atoms here, since there
    should be one on the calculator.
    '''
    if atoms is not None:
            self.atoms = atoms.copy()
    self.write_input(self.atoms, properties, system_changes)
    
    olddir = os.getcwd()
    try:
        os.chdir(self.directory)
        errorcode = self.run()
    finally:
        os.chdir(olddir)

    if errorcode:
        raise RuntimeError('%s returned an error: %d' %
                           (self.name, errorcode))
    self.read_results()
   

Aims.calculate = calculate


def run(self):
    '''monkey patch to submit job through the queue.

    If this is called, then the calculator thinks a job should be run.
    If we are in the queue, we should run it, otherwise, a job should
    be submitted.

    '''

    # if we are in the queue and xaims is called or if we want to use
    # mode='run' , we should just run the job. First, we consider how.
    if 'PBS_O_WORKDIR' in os.environ or XAIMSRC['mode'] == 'run':
        log.info('In the queue. determining how to run')
        if 'PBS_NODEFILE' in os.environ:
            # we are in the queue. determine if we should run serial
            # or parallel
            NPROCS = len(open(os.environ['PBS_NODEFILE']).readlines())
            log.debug('Found {0} PROCS'.format(NPROCS))
            if NPROCS == 1:
                # no question. running in serial.
                aimscmd = XAIMSRC['aims.executable.serial']
                log.debug('NPROCS = 1. running in serial')
                exitcode = os.system(aimscmd)
                return exitcode
            else:
                # vanilla MPI run. multiprocessing does not work on more
                # than one node, and you must specify in XAIMSRC to use it
                if (XAIMSRC['queue.nodes'] > 1
                    or (XAIMSRC['queue.nodes'] == 1
                        and XAIMSRC['queue.ppn'] > 1
                        and (XAIMSRC['multiprocessing.cores_per_process']
                             == 'None'))):
                    log.debug('queue.nodes = {0}'.format(XAIMSRC['queue.nodes']))
                    log.debug('queue.ppn = {0}'.format(XAIMSRC['queue.ppn']))
                    log.debug('multiprocessing.cores_per_process'
                              '= {0}'.format(XAIMSRC['multiprocessing.cores_per_process']))
                    log.debug('running vanilla MPI job')

                    print('MPI NPROCS = ', NPROCS)
                    aimscmd = XAIMSRC['aims.executable.parallel']
                    parcmd = 'mpirun -np %i %s' % (NPROCS, aimscmd)
                    exitcode = os.system(parcmd)
                    return exitcode
                else:
                    # we need to run an MPI job on cores_per_process
                    if XAIMSRC['multiprocessing.cores_per_process'] == 1:
                        log.debug('running single core multiprocessing job')
                        aimscmd = XAIMSRC['aims.executable.serial']
                        exitcode = os.system(aimscmd)
                    elif XAIMSRC['multiprocessing.cores_per_process'] > 1:
                        log.debug('running mpi multiprocessing job')
                        NPROCS = XAIMSRC['multiprocessing.cores_per_process']

                        aimscmd = XAIMSRC['aims.executable.parallel']
                        parcmd = 'mpirun -np %i %s' % (NPROCS, aimscmd)
                        exitcode = os.system(parcmd)
                        return exitcode
        else:
            # probably running at cmd line, in serial.
            aimscmd = XAIMSRC['aims.executable.serial']
            exitcode = os.system(aimscmd)
            return exitcode
        # end

    # if you get here, a job is getting submitted
    script = '''
#!/bin/bash
cd {self.cwd}  # this is the current working directory
cd {self.aimsdir}  # this is the aims directory
runxaims.py     # this is the aims command
#end'''.format(**locals())

    jobname = self.aimsdir
    log.debug('{0} will be the jobname.'.format(jobname))
    log.debug('-l nodes={0}:ppn={1}'.format(XAIMSRC['queue.nodes'],
                                            XAIMSRC['queue.ppn']))

    cmdlist = ['{0}'.format(XAIMSRC['queue.command'])]
    cmdlist += [option for option in XAIMSRC['queue.options'].split()]
    cmdlist += ['-N', '{0}'.format(jobname),
                '-l walltime={0}'.format(XAIMSRC['queue.walltime']),
                '-l nodes={0}:ppn={1}'.format(XAIMSRC['queue.nodes'],
                                              XAIMSRC['queue.ppn']),
                '-l mem={0}'.format(XAIMSRC['queue.mem'])]
    log.debug('{0}'.format(' '.join(cmdlist)))
    p = Popen(cmdlist,
              stdin=PIPE, stdout=PIPE, stderr=PIPE)

    log.debug(script)

    out, err = p.communicate(script)

    if out == '' or err != '':
        raise Exception('something went wrong in qsub:\n\n{0}'.format(err))

    f = open('jobid', 'w')
    f.write(out)
    f.close()

    raise aimsSubmitted(out)

Aims.run = run


