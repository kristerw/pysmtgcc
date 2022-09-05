import gcc
from z3 import *

import smtgcc

VERBOSE = 0


class TranslationValidation(gcc.GimplePass):
    def __init__(self):
        self.functions = {}
        self.state = None
        gcc.GimplePass.__init__(self, name="TranslationValidation")

    def execute(self, fun):
        try:
            if fun.decl.name not in ["src", "tgt"]:
                raise smtgcc.Error("Invalid function name", fun.decl.location)
            if not self.state:
                self.state = smtgcc.init_common_state(fun)
            smt_fun = smtgcc.process_function(fun, self.state, False)
            self.functions[smt_fun.name] = smt_fun
            if "src" in self.functions and "tgt" in self.functions:
                src_smt = self.functions["src"]
                tgt_smt = self.functions["tgt"]
                smtgcc.check(src_smt, tgt_smt, fun.decl.location, True, VERBOSE)
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


tv = TranslationValidation()
tv.register_after("ssa")
