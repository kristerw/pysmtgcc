import gcc
from z3 import *

import smtgcc

VERBOSE = 0


def tv_callback(*args, **kwargs):
    (opt_pass, fun, tv) = args
    if isinstance(opt_pass, (gcc.IpaPass, gcc.SimpleIpaPass)):
        # IPA passes implements translations that we will report as
        # incorrect when we only analyze one function at a time.
        # It would be good to iterate over the function to check the last
        # pass before IPA, but I cannot find any relevant API in the python
        # plugin.
        for name in tv.functions:
            function = tv.functions[name]
            function.smt_fun = None
            function.next_pass_name = "ipa"
        return
    if isinstance(opt_pass, gcc.RtlPass):
        # We do not do any checking for RTL passes.
        return
    assert isinstance(opt_pass, gcc.GimplePass)

    if fun.decl.name not in tv.functions:
        tv.functions[fun.decl.name] = Function()

    function = tv.functions[fun.decl.name]
    if not function.in_ssa_form:
        if opt_pass.name == "ssa":
            function.in_ssa_form = True
            function.next_pass_name = opt_pass.name
        return
    if opt_pass.name[0] == "*":
        return

    if function.next_pass_name in ["ccp", "isolate-paths"]:
        # This is a pass that may change the IR in a way that makes us
        # (incorrectly) believe the transformation is invalid.
        function.smt_fun = None

    try:
        if function.smt_fun is not None:
            src_smt_fun = function.smt_fun
            src_pass_name = function.pass_name
            tgt_pass_name = function.next_pass_name
            transform_name = src_pass_name + " -> " + tgt_pass_name
            tgt_smt_fun = smtgcc.process_function(fun, function.state, True)
            smtgcc.check(
                src_smt_fun,
                tgt_smt_fun,
                fun.decl.location,
                False,
                VERBOSE,
                transform_name,
            )

        function.state = smtgcc.init_common_state(fun)
        function.smt_fun = smtgcc.process_function(fun, function.state, False)
        function.pass_name = function.next_pass_name
        function.next_pass_name = opt_pass.name
    except (smtgcc.Error, smtgcc.NotImplementedError) as err:
        msg = str(err)
        function = tv.functions[fun.decl.name]
        if msg not in function.errors:
            function.errors.append(msg)
            if err.location:
                location = err.location
            else:
                location = fun.decl.location

            if isinstance(err, smtgcc.NotImplementedError):
                msg = "Not implemented: " + msg
            else:
                msg = "Error: " + msg
            gcc.inform(location, msg)
        function.smt_fun = None
        function.next_pass_name = opt_pass.name


class Function:
    def __init__(self):
        self.in_ssa_form = False
        self.state = None
        self.smt_fun = None
        self.pass_name = None
        self.next_pass_name = None
        self.errors = []


class TranslationValidation:
    def __init__(self):
        self.functions = {}


tv = TranslationValidation()
gcc.register_callback(gcc.PLUGIN_PASS_EXECUTION, tv_callback, (tv,))
