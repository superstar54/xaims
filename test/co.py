#!/usr/bin/env python
from ase import Atoms, Atom
from ase.calculators.aims import Aims
from ase.calculators.xaims import *
import ase.units
atoms = Atoms('CO', ([0, 0, 0], [0, 0, 1.2]))
with Xaims(model=None,
	       label='co-test',
           xc='PBE',
           sc_accuracy_etot=1e-5,
           sc_accuracy_eev=1e-3,
           sc_accuracy_rho=1e-4,
           sc_iter_limit=40,
           atoms=atoms) as calc:
    atoms.set_calculator(calc)
    print('energy = {0} eV'.format(atoms.get_potential_energy()))
