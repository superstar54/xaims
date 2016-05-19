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





class Xaims:
    '''Context manager for running Aims calculations

    On entering, automatically change to working aims directory, and
    on exit, automatically change back to original working directory.

    Note: You do not want to raise exceptions here! it makes code
    using this really hard to write because you have to catch
    exceptions in the with statement.
    '''

    def __init__(self, vib = None, **kwargs):
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
        if vib==True:
            pass
            #self.git_vibrations()
        else:
        # and get the new calculator
            try:
                calc = Aims(**self.kwargs)
                return calc
            except:
                self.__exit__()
                raise

    def __exit__(self, *args):
        '''
        on exit, change back to the original directory.
        '''

        return False  # allows exception to propagate out

