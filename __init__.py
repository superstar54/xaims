"""Wrapper module ase.calculators.aims to enable automatic job handling with the
Torque queue system.


Xaims(**kwargs) returns a monkeypatched Aims calculator.

xaims is a context manager that creates dir necessary, changes into it, does
stuff, and changes back to the original directory when done.

>>> with xaims(dir, **kwargs) as calc:
...     do stuff


Find the source at http://github.com/superstar54/xaims 

"""

from jasp import *
