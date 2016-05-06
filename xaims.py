#!/usr/bin/env python
'''this is a patched :mod:`ase.calculators.aims.Aims` calculator

with the following features:

1. context manager to run in a specified directory and then return to the CWD.
2. calculations are run through the queue, not at the command line.
3. hook functions are enabled for pre and post processing
4. atoms is now a keyword
'''

import os
import sys
from hashlib import sha1
from subprocess import Popen, PIPE

# this is used in xaims_extensions. We will have to remove it at some point.
import commands

from ase import Atoms
from ase.calculators.aims import *

import numpy as np
np.set_printoptions(precision=3, suppress=True)

# internal imports
from xaimsrc import *          # configuration data

from xaims_exceptions import *  # exception definitions
from xaims_extensions import *  # extensions to aims.py

# ###################################################################
# Logger for handling information, warning and debugging
# ###################################################################
import logging
log = logging.getLogger('Xaims')
log.setLevel(logging.CRITICAL)
handler = logging.StreamHandler()
if sys.version_info < (2, 5):  # no funcName in python 2.4
    formatstring = ('%(levelname)-10s '
                    'lineno: %(lineno)-4d %(message)s')
else:
    formatstring = ('%(levelname)-10s function: %(funcName)s '
                    'lineno: %(lineno)-4d %(message)s')
formatter = logging.Formatter(formatstring)
handler.setFormatter(formatter)
log.addHandler(handler)

# * Utility functions
# ** Calculation is ok
def calculation_is_ok():
    '''Returns bool if calculation appears ok.

    That means:
    1. There is a aims.out with contents
    '''
    with open('aims.out') as f:
        converged = False
        lines = f.readlines()
        for n, line in enumerate(lines):
            if line.rfind('Have a nice day') > -1:
                converged = True
    return converged


# * Xaims
# ###################################################################
# Xaims function - returns a Aims calculator
# ###################################################################
Aims.results = {}  # for storing data used in ase.db
Aims.name = 'xaims'

# I thought of setting defaults like this. But, I realized it would
# break reading old calculations, where some of these are not set. I
# am leaving this in for now.
default_parameters = {'xc': 'PBE',
                      'kpts': (1, 1, 1)}


def compatible_atoms_p(a1, a2):
    '''Returns whether atoms have changed from what went in to xaims to what was
read. we only care if the number or types of atoms changed.
    a1 is the directory atoms
    a2 is the passed atoms.'''
    log.debug('Checking if {0} compatible with {1}.'.format(a1, a2))
    if ((len(a1) != len(a2))
        or
        (a1.get_chemical_symbols() != a2.get_chemical_symbols())):
        raise Exception('Incompatible atoms.\n'
                        '{0} contains {1}'
                        ' but you passed {2}, which is not '
                        'compatible'.format(os.getcwd(),
                                            a1, a2))


def Xaims(debug=10,
         restart=None,
         output_template='aims',
         track_output=False,
         atoms=None,
         label=None,
         **kwargs):
    '''wrapper function to create a Aims calculator. The only purpose
    of this function is to enable atoms as a keyword argument, and to
    restart the calculator from the current directory if no keywords
    are given.

    By default we delete these large files. We do not need them very
    often, so the default is to delete them, and only keep them when
    we know we want them.

    **kwargs is the same as ase.calculators.aims.

    you must be in the directory where aims will be run.

    '''

    if debug is not None:
        log.setLevel(debug)

    log.debug('Xaims called in %s', os.getcwd())
    log.debug('kwargs = %s', kwargs)

# ** Empty directory starting from scratch
    # empty aims dir. start from scratch
    if (not os.path.exists('aims.out')):
        calc = Aims(label=label, **kwargs)

        if atoms is not None:
            atoms.calc = calc
        log.debug('empty aims dir. start from scratch')

# ** initialized directory, but no job has been run
    elif (os.path.exists('aims.out')
          # but not converged
          and not calculation_is_ok()):
        log.debug('initialized directory, but no job has been run')

        calc = Aims(label=label, **kwargs)
        if atoms is not None:
            atoms.calc = calc

# ** job is created, not in queue, not running. finished and first time we are looking at it
    elif (os.path.exists('aims.out')
          and calculation_is_ok()):
        log.debug('job is created, not in queue, not running.'
                  'finished and first time we are looking at it')
        
        calc = Aims(restart=True, label=label, **kwargs)  # automatically loads results

    else:
        raise AimsUnknownState('I do not recognize the state of this'
                               'directory {0}'.format(os.getcwd()))

    return calc


class cd:
    '''Context manager for changing directories.

    On entering, store initial location, change to the desired directory,
    creating it if needed.  On exit, change back to the original directory.

    Example:
    with cd('path/to/a/calculation'):
        calc = xaims(args)
        calc.get_potential energy()
    '''

    def __init__(self, working_directory):
        self.origin = os.getcwd()
        self.wd = working_directory


    def __enter__(self):
        # make directory if it doesn't already exist
        if not os.path.isdir(self.wd):
            os.makedirs(self.wd)

        # now change to new working dir
        os.chdir(self.wd)


    def __exit__(self, *args):
        os.chdir(self.origin)
        return False # allows body exceptions to propagate out.


class xaims:
    '''Context manager for running Aims calculations

    On entering, automatically change to working aims directory, and
    on exit, automatically change back to original working directory.

    Note: You do not want to raise exceptions here! it makes code
    using this really hard to write because you have to catch
    exceptions in the with statement.
    '''

    def __init__(self, aimsdir, **kwargs):
        '''
        aimsdir: the directory to run aims in

        **kwargs: all the aims keywords, including an atoms object
        '''

        self.cwd = os.getcwd()  # directory we were in when xaims created
        self.aimsdir = os.path.expanduser(aimsdir)

        self.kwargs = kwargs  # this does not include the aimsdir variable

    def __enter__(self):
        '''
        on enter, make sure directory exists, create it if necessary,
        and change into the directory. then return the calculator.

        try not to raise exceptions in here to avoid needing code like:
        try:
            with xaims() as calc:
                do stuff
        except:
            do stuff.

        I want this syntax:
        with xaims() as calc:
            try:
                calc.do something
            except (aimsException):
                do something.
        '''
        # make directory if it doesn't already exist
        if not os.path.isdir(self.aimsdir):
            os.makedirs(self.aimsdir)

        # now change to new working dir
        os.chdir(self.aimsdir)

        # and get the new calculator
        try:
            calc = Xaims(label=self.aimsdir, **self.kwargs)
            calc.aimsdir = self.aimsdir   # aims directory
            calc.cwd = self.cwd   # directory we came from
            os.chdir(self.cwd)
            return calc
        except:
            self.__exit__()
            raise

    def __exit__(self, *args):
        '''
        on exit, change back to the original directory.
        '''
        os.chdir(self.cwd)
        return False  # allows exception to propagate out


def isaaimsdir(path):
    '''Return bool if the current working directory is a aims directory.

    A aims dir has the aims files in it. This function is typically used
    when walking a filesystem to identify directories that contain
    calculation results.
    '''
    # standard aimsdir
    if (os.path.exists(os.path.join(path, 'control.in')) and
        os.path.exists(os.path.join(path, 'geometry.in'))):
        return True
    else:
        return False

