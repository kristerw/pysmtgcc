import gcc
from z3 import *

import smtgcc

VERBOSE = 0


class FindUB(gcc.GimplePass):
    def __init__(self):
        gcc.GimplePass.__init__(self, name="FindUB")

    def execute(self, fun):
        try:
            state = smtgcc.init_common_state(fun)
            smt_fun = smtgcc.process_function(fun, state, False)
            smtgcc.find_ub(smt_fun, fun.decl.location, VERBOSE)
        except (smtgcc.Error, smtgcc.NotImplementedError) as err:
            if err.location:
                location = err.location
            else:
                location = fun.decl.location
            if isinstance(err, smtgcc.NotImplementedError):
                msg = "Not implemented: " + str(err)
            else:
                msg = "Error: " + str(err)
            gcc.inform(location, msg)


find_ub = FindUB()
find_ub.register_after("ssa")
