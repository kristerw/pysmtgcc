import gcc
from z3 import *


# Solver memory limit (in megabytes) or 0 for "no limit".
SOLVER_MAX_MEMORY = 25 * 1024

# Solver timeout (in seconds) or 0 for "no limit".
SOLVER_TIMEOUT = 5 * 60

# Size of anonymous memory size blocks we may need to introduce (for example,
# so that function pointer arguments have memory to point to).
ANON_MEM_SIZE = 64

# How many bytes load, store, __builtin_memset, etc. can expand.
MAX_MEMORY_UNROLL_LIMIT = 10000

# How the pointer bits are divided between id and offset.
PTR_OFFSET_BITS = 40
PTR_ID_BITS = 24
assert PTR_ID_BITS + PTR_OFFSET_BITS == 64


class NotImplementedError(Exception):
    """Raised when encountering unimplemented constructs."""

    def __init__(self, string, location=None):
        self.location = location
        Exception.__init__(self, string)


class Error(Exception):
    """Raised when being passed incorrect input, etc."""

    def __init__(self, string, location=None):
        self.location = location
        Exception.__init__(self, string)


class CommonState:
    """Things that are common for the source and target functions."""

    def __init__(
        self,
        global_memory,
        decl_to_memory,
        const_mem_ids,
        next_id,
        ptr_constraints,
        memory,
        mem_sizes,
        is_initialized,
        fun_args,
    ):
        self.global_memory = global_memory
        self.const_mem_ids = const_mem_ids
        self.local_memory = None
        self.next_id = next_id
        self.ptr_constraints = ptr_constraints
        self.memory = memory
        self.is_initialized = is_initialized
        self.mem_sizes = mem_sizes
        self.decl_to_memory = decl_to_memory
        self.fun_args = fun_args
        self.retval = None


class MemoryBlock:
    def __init__(self, decl, size, mem_id):
        if size > (1 << PTR_OFFSET_BITS):
            if decl is not None:
                location = decl.location
            else:
                location = None
            raise NotImplementedError("Too big memory block", location)
        self.size = size
        self.decl = decl
        self.mem_id = mem_id
        name = ".mem_" + str(self.mem_id)
        self.mem_array = Array(name, BitVecSort(PTR_OFFSET_BITS), BitVecSort(8))


class SmtPointer:
    def __init__(self, mem_id, offset):
        assert mem_id.sort() == BitVecSort(PTR_ID_BITS)
        assert offset.sort() == BitVecSort(PTR_OFFSET_BITS)
        self.mem_id = mem_id
        self.offset = offset

    def bitvector(self):
        return Concat([self.mem_id, self.offset])


class SmtFunction:
    def __init__(self, fun, state):
        self.fun = fun
        self.name = fun.decl.name
        self.tree_to_smt = {}
        self.tree_is_initialized = {}
        self.decl_to_id = {}
        self.bb2smt = {}
        self.state = state

        # Function return
        self.retval = None
        self.invokes_ub = None
        self.memory = None
        self.mem_sizes = None
        self.is_initialized = None

    def add_ub(self, smt):
        if self.invokes_ub is None:
            self.invokes_ub = smt
        else:
            self.invokes_ub = Or(self.invokes_ub, smt)


class SmtBB:
    def __init__(self, bb, smt_fun, mem_sizes):
        self.bb = bb
        self.smt_fun = smt_fun
        self.invokes_ub = None
        self.smtcond = None
        self.retval = None
        self.labels = self._find_labels()
        self.switch_label_to_cond = None
        smt_fun.bb2smt[bb] = self

        if len(bb.preds) == 0:
            self.memory = smt_fun.state.memory
            self.mem_sizes = mem_sizes
            self.is_initialized = smt_fun.state.is_initialized
        else:
            memory = []
            mem_sizes = []
            is_initialized = []
            for edge in bb.preds:
                src_smt_bb = smt_fun.bb2smt[edge.src]
                memory.append((src_smt_bb.memory, edge))
                mem_sizes.append((src_smt_bb.mem_sizes, edge))
                is_initialized.append((src_smt_bb.is_initialized, edge))
            self.memory = process_phi_smt_args(memory, smt_fun)
            self.mem_sizes = process_phi_smt_args(mem_sizes, smt_fun)
            self.is_initialized = process_phi_smt_args(is_initialized, smt_fun)

        # Create condition telling if this BB is executed or not.
        if len(bb.preds) == 0:
            # The entry block is always executed.
            self.is_executed = BoolVal(True)
        else:
            # The block is executed if any of the edges are executed.
            is_executed = None
            for edge in bb.preds:
                cond = build_full_edge_cond(edge, smt_fun)
                if is_executed is None:
                    is_executed = cond
                else:
                    is_executed = Or(is_executed, cond)
            self.is_executed = is_executed

    def _find_labels(self):
        labels = []
        for stmt in self.bb.gimple:
            if isinstance(stmt, gcc.GimpleLabel):
                labels.append(stmt.label)
        return labels

    def add_ub(self, smt):
        if self.invokes_ub is None:
            self.invokes_ub = smt
        else:
            self.invokes_ub = Or(self.invokes_ub, smt)


def build_switch_label_to_cond(stmt, smt_bb):
    "Build SMT conditions corresponding to the case(s) jumping to each label."
    assert isinstance(stmt, gcc.GimpleSwitch)
    assert isinstance(stmt.indexvar.type, gcc.IntegerType)
    label_to_cond = {}
    default_cond = None
    indexvar = get_tree_as_smt(stmt.indexvar, smt_bb)
    for expr in stmt.labels[1:]:
        low = get_tree_as_smt(expr.low, smt_bb)

        # The case type may be of a lower precision. If so, extend it.
        if stmt.indexvar.type.precision > expr.low.type.precision:
            extbits = stmt.indexvar.type.precision - expr.low.type.precision
            if expr.low.type.unsigned:
                low = ZeroExt(extbits, low)
            else:
                low = SignExt(extbits, low)

        if expr.high is not None:
            high = get_tree_as_smt(expr.high, smt_bb)
            if stmt.indexvar.type.precision > expr.high.type.precision:
                extbits = stmt.indexvar.type.precision - expr.high.type.precision
                if expr.high.type.unsigned:
                    high = ZeroExt(extbits, high)
                else:
                    high = SignExt(extbits, high)

            if stmt.indexvar.type.unsigned:
                low_cond = ULE(low, indexvar)
                high_cond = ULE(indexvar, high)
            else:
                low_cond = low <= indexvar
                high_cond = indexvar <= high
            cond = And(low_cond, high_cond)
        else:
            cond = indexvar == low

        label = expr.target
        if label in label_to_cond:
            label_to_cond[label] = Or(label_to_cond[label], cond)
        else:
            label_to_cond[label] = cond

        if default_cond is None:
            default_cond = Not(cond)
        else:
            default_cond = And(default_cond, Not(cond))

    default_label = stmt.labels[0].target
    if default_label in label_to_cond:
        label_to_cond[default_label] = Or(label_to_cond[default_label], default_cond)
    else:
        label_to_cond[default_label] = default_cond

    return label_to_cond


def get_smt_sort(type):
    """Return the SMT sort for the GCC type."""
    if isinstance(type, gcc.IntegerType):
        return BitVecSort(type.precision)
    if isinstance(type, gcc.BooleanType):
        return BoolSort()
    if isinstance(type, gcc.RealType):
        if type.precision == 16:
            return Float16()
        if type.precision == 32:
            return Float32()
        if type.precision == 64:
            return Float64()
        if type.precision == 80:
            # 80-bit floats does not follow the IEEE standard, so this is
            # "wrong". But GCC does not do any optimizations on GIMPLE that
            # depend on the properties that differ, so this is OK for now.
            return FPSort(15, 64)
        if type.precision == 128:
            return Float128()

    raise NotImplementedError(f"get_smt_sort {type.__class__}")


def load_bytes(mem_id, offset, size, smt_bb):
    assert size > 0
    if size > MAX_MEMORY_UNROLL_LIMIT:
        raise NotImplementedError(f"load_bytes too big ({size})")

    out_of_bound_ub_check(mem_id, offset, size, smt_bb)

    # Load the "byte" values from memory.
    bytes = []
    is_initialized = []
    ptr = Concat([mem_id, offset])
    if is_bv_value(mem_id) and is_bv_value(offset):
        ptr = simplify(ptr)
    for i in range(0, size):
        p = ptr + i
        if is_bv_value(ptr):
            p = simplify(p)
        bytes.insert(0, Select(smt_bb.memory, p))
        is_initialized.append(Select(smt_bb.is_initialized, p))

    # Construct a bit-vector for the loaded value.
    if size == 1:
        result = bytes[0]
    else:
        result = Concat(bytes)

    return result, is_initialized


def load_value(expr, smt_bb):
    mem_id, offset = build_smt_addr(expr, smt_bb)
    result, is_initialized = load_bytes(mem_id, offset, expr.type.sizeof, smt_bb)

    # Convert the bit-vector to the correct type.
    if isinstance(expr.type, gcc.RealType):
        if expr.type.precision == 80:
            result = Extract(78, 0, result)
        result = fpBVToFP(result, get_smt_sort(expr.type))
    elif isinstance(expr.type, gcc.BooleanType):
        result = result != 0
    elif isinstance(expr.type, gcc.PointerType):
        mem_id = Extract(63, PTR_OFFSET_BITS, result)
        offset = Extract(PTR_OFFSET_BITS - 1, 0, result)
        result = SmtPointer(mem_id, offset)
    elif not isinstance(expr.type, gcc.IntegerType):
        raise NotImplementedError("load " + str(expr.type.__class__))

    return result, is_initialized


def load_bitfield(expr, smt_bb):
    assert is_bitfield(expr)
    size = (expr.bitoffset + expr.type.precision + 7) // 8
    mem_id, offset = build_smt_addr(expr, smt_bb)
    bytes, is_initialized = load_bytes(mem_id, offset, size, smt_bb)
    result = Extract(expr.bitoffset + expr.type.precision - 1, expr.bitoffset, bytes)
    # The value in is_initialized may be of the wrong size; for example, when
    # loading a 16-bit value (2 bytes) from a misaligned 16-bit bitfield
    # (3 bytes).
    if len(is_initialized) == 1:
        is_init = is_initialized[0]
    else:
        is_init = And(is_initialized)
    is_initialized = [is_init] * ((expr.type.precision + 7) // 8)
    return result, is_initialized


def load(stmt, smt_bb):
    if isinstance(stmt, gcc.GimpleReturn):
        expr = stmt.retval
    else:
        expr = stmt.rhs[0]
    if is_bitfield(expr):
        res, is_initialized = load_bitfield(expr, smt_bb)
    else:
        res, is_initialized = load_value(expr, smt_bb)
    return res, is_initialized


def out_of_bound_ub_check(mem_id, offset, size, smt_bb):
    is_valid_mem_id = UGE(mem_id, smt_bb.smt_fun.state.next_id)
    if is_bv_value(mem_id):
        is_valid_mem_id = simplify(is_valid_mem_id)
        if is_true(is_valid_mem_id):
            smt_bb.add_ub(is_valid_mem_id)
        else:
            assert is_false(is_valid_mem_id)
    else:
        smt_bb.add_ub(is_valid_mem_id)

    smt_size = Select(smt_bb.mem_sizes, mem_id)
    if is_bv_value(mem_id):
        smt_size = simplify(smt_size)
    is_out_of_bound = Or(UGT(offset + size, smt_size), UGE(offset, offset + size))
    if is_bv_value(smt_size) and is_bv_value(offset):
        is_out_of_bound = simplify(is_out_of_bound)
        if is_true(is_out_of_bound):
            smt_bb.add_ub(is_out_of_bound)
        else:
            assert is_false(is_out_of_bound)
    else:
        smt_bb.add_ub(is_out_of_bound)


def const_mem_ub_check(mem_id, smt_bb):
    const_mem_ids = smt_bb.smt_fun.state.const_mem_ids
    if const_mem_ids:
        is_ub = mem_id == const_mem_ids[0]
        for cmem_id in const_mem_ids[1:]:
            is_ub = Or(is_ub, mem_id == cmem_id)
        if is_bv_value(mem_id):
            is_ub = simplify(is_ub)
        smt_bb.add_ub(is_ub)


def store_bytes(mem_id, offset, size, value, is_initialized, smt_bb):
    assert size > 0
    assert len(is_initialized) == size
    if size > MAX_MEMORY_UNROLL_LIMIT:
        raise NotImplementedError(f"store_bytes too big ({size})")

    out_of_bound_ub_check(mem_id, offset, size, smt_bb)
    const_mem_ub_check(mem_id, smt_bb)

    ptr = Concat([mem_id, offset])
    if is_bv_value(mem_id) and is_bv_value(offset):
        ptr = simplify(ptr)
    for i in range(0, size):
        byte_value = Extract(i * 8 + 7, i * 8, value)
        if is_bv_value(value):
            byte_value = simplify(byte_value)
        p = ptr + i
        if is_bv_value(ptr):
            p = simplify(p)
        smt_bb.memory = Store(smt_bb.memory, p, byte_value)
        smt_bb.is_initialized = Store(smt_bb.is_initialized, p, is_initialized[i])


def store_value(expr, value, smt_bb):
    # Create a bit-vector for the value to store.
    type = expr.type
    if isinstance(type, gcc.RealType):
        value = fpToIEEEBV(value)
        if type.precision == 80:
            value = Concat([BitVecVal(0, 49), value])
    elif isinstance(type, gcc.BooleanType):
        bv0 = BitVecVal(0, 8)
        bv1 = BitVecVal(1, 8)
        value = If(value, bv1, bv0)
    elif isinstance(type, gcc.PointerType):
        value = value.bitvector()
    elif not isinstance(type, gcc.IntegerType):
        raise NotImplementedError("store " + str(type.__class__))

    mem_id, offset = build_smt_addr(expr, smt_bb)
    if expr in smt_bb.smt_fun.tree_is_initialized:
        is_initialized = smt_bb.smt_fun.tree_is_initialized[expr]
    else:
        is_initialized = [BoolVal(True)] * type.sizeof
    store_bytes(mem_id, offset, type.sizeof, value, is_initialized, smt_bb)


def store_bitfield(expr, value, smt_bb):
    assert is_bitfield(expr)
    mem_id, offset = build_smt_addr(expr, smt_bb)

    if is_bool(value):
        assert expr.type.precision == 1
        value = If(value, BitVecVal(1, 1), BitVecVal(0, 1))

    size = (expr.bitoffset + expr.type.precision + 7) // 8
    result, _ = load_bytes(mem_id, offset, size, smt_bb)
    parts = []
    if expr.bitoffset + expr.type.precision != size * 8:
        nof_bits = size * 8
        parts.append(
            Extract(nof_bits - 1, expr.bitoffset + expr.type.precision, result)
        )
    parts.append(value)
    if expr.bitoffset != 0:
        parts.append(Extract(expr.bitoffset - 1, 0, result))
    if len(parts) == 1:
        value = parts[0]
    else:
        value = Concat(parts)

    if expr in smt_bb.smt_fun.tree_is_initialized:
        is_initialized = smt_bb.smt_fun.tree_is_initialized[expr]
        # The value store in tree_is_initialized may be of the wrong size;
        # for example, when storing a 16-bit value (2 bytes) in a misaligned
        # 16-bit bitfield (3 bytes).
        if len(is_initialized) == 1:
            is_init = is_initialized[0]
        else:
            is_init = And(is_initialized)
        is_initialized = [is_init] * size
    else:
        is_initialized = [BoolVal(True)] * size
    store_bytes(mem_id, offset, size, value, is_initialized, smt_bb)


def store(stmt, smt_bb):
    if stmt.exprcode == gcc.Constructor:
        if not (stmt.rhs[0].is_clobber or len(stmt.rhs[0].elements) == 0):
            raise NotImplementedError("store constructor that is not a clobber")
        size = stmt.rhs[0].type.sizeof
        if size == 0:
            # This happens for {CLOBBER(eol) constructors for zero
            # sized arrays such as "int x[0];".
            return
        if size > MAX_MEMORY_UNROLL_LIMIT:
            raise NotImplementedError(f"store large constructor ({size} bytes)")
        mem_id, smt_offset = build_smt_addr(stmt.lhs, smt_bb)
        if stmt.rhs[0].is_clobber:
            smt_bb.is_initialized = mark_mem_uninitialized(
                mem_id, smt_offset, size, smt_bb.is_initialized
            )
        else:
            is_initialized = [BoolVal(True)] * size
            rhs = BitVecVal(0, size * 8)
            store_bytes(mem_id, smt_offset, size, rhs, is_initialized, smt_bb)
        return

    value = process_unary(stmt, smt_bb, False)
    if is_bitfield(stmt.lhs):
        store_bitfield(stmt.lhs, value, smt_bb)
    else:
        store_value(stmt.lhs, value, smt_bb)


def has_loop(fun):
    "Does the function contain a loop?"
    seen_bbs = []
    for bb in fun.cfg.inverted_post_order:
        seen_bbs.append(bb)
        for succ in bb.succs:
            if succ.dest in seen_bbs:
                return True
    return False


def uninit_var_to_smt(expr):
    "Build a Const representing read from an uninitialized variable."
    # Variables generated by the compiler gets None as expr.var.name.
    # So generate the Const name by using str(expr.var) instead.
    name = f".undef.{expr.var}"
    if isinstance(expr.type, gcc.PointerType):
        ptr = Const(name, BitVecSort(64))
        mem_id = Extract(63, PTR_OFFSET_BITS, ptr)
        offset = Extract(PTR_OFFSET_BITS - 1, 0, ptr)
        return SmtPointer(mem_id, offset)
    return Const(name, get_smt_sort(expr.type))


def get_tree_as_smt(expr, smt_bb, uninit_is_ub=True):
    if uninit_is_ub and expr in smt_bb.smt_fun.tree_is_initialized:
        is_initialized = smt_bb.smt_fun.tree_is_initialized[expr]
        if len(is_initialized) == 1:
            is_uninitialized = Not(is_initialized[0])
        else:
            is_uninitialized = Not(And(is_initialized))
        smt_bb.add_ub(is_uninitialized)

    if isinstance(expr, gcc.ParmDecl):
        return smt_bb.smt_fun.tree_to_smt[expr]
    if isinstance(expr, gcc.SsaName):
        if expr in smt_bb.smt_fun.tree_to_smt:
            return smt_bb.smt_fun.tree_to_smt[expr]
        if isinstance(expr.var, gcc.ParmDecl):
            if (
                expr.var not in smt_bb.smt_fun.tree_to_smt
                and expr.var.is_artificial
                and expr.var.name[:5] == "CHAIN"
            ):
                raise NotImplementedError(f"get_tree_as_smt nested functions")
            return smt_bb.smt_fun.tree_to_smt[expr.var]
        if isinstance(expr.var, gcc.VarDecl):
            # We are reading from an uninitialized local variable.
            # Mark this as uninitialized, and create a new unconstrained
            # constant to make the generated SMT typecheck.
            if uninit_is_ub:
                smt_bb.add_ub(BoolVal(True))
            else:
                is_initialized = [BoolVal(False)] * expr.var.type.sizeof
                smt_bb.smt_fun.tree_is_initialized[expr] = is_initialized
            return uninit_var_to_smt(expr)
        raise NotImplementedError(f"get_tree_as_smt SsaName {expr}")
    if isinstance(expr, gcc.IntegerCst):
        if isinstance(expr.type, gcc.BooleanType):
            return BoolVal(expr.constant != 0)
        if isinstance(expr.type, gcc.IntegerType):
            return BitVecVal(expr.constant, expr.type.precision)
        if isinstance(expr.type, gcc.PointerType):
            cst = BitVecVal(expr.constant, 64)
            mem_id = Extract(63, PTR_OFFSET_BITS, cst)
            offset = Extract(PTR_OFFSET_BITS - 1, 0, cst)
            return SmtPointer(mem_id, offset)
    if isinstance(expr, gcc.ViewConvertExpr):
        # Ignore uninit_is_ub as we are doing an operation on the value,
        # which is UB if iti is uninitialized.
        value = get_tree_as_smt(expr.operand, smt_bb, True)
        return bit_cast(value, expr.operand.type, expr.type)
    if isinstance(expr, gcc.RealCst):
        return FPVal(expr.constant, get_smt_sort(expr.type))
    if isinstance(expr, gcc.AddrExpr):
        mem_id, offset = build_smt_addr(expr.operand, smt_bb)
        return SmtPointer(mem_id, offset)

    raise NotImplementedError(f"get_tree_as_smt {expr.__class__} {expr.type.__class__}")


def is_bitfield(expr):
    "Return true if expr is a gcc.ComponentRef for a bitfield."
    if not isinstance(expr, gcc.ComponentRef):
        return False
    if not isinstance(expr.type, gcc.IntegerType):
        return False
    if expr.bitoffset % 8 != 0:
        return True
    # If a bit field with a size that is a multiple of a byte, and the field
    # is byte aligned, then we can treat it as a normal load/store.
    return expr.type.precision != expr.type.sizeof * 8


def process_comparison(stmt, smt_bb):
    if isinstance(stmt, gcc.GimpleCond):
        type = stmt.lhs.type
        rhs0 = get_tree_as_smt(stmt.lhs, smt_bb)
        rhs1 = get_tree_as_smt(stmt.rhs, smt_bb)
    else:
        assert isinstance(stmt, gcc.GimpleAssign)
        type = stmt.rhs[0].type
        rhs0 = get_tree_as_smt(stmt.rhs[0], smt_bb)
        rhs1 = get_tree_as_smt(stmt.rhs[1], smt_bb)
    if isinstance(type, gcc.IntegerType):
        is_unsigned = type.unsigned
    elif isinstance(type, gcc.RealType):
        pass
    elif isinstance(type, gcc.BooleanType):
        if stmt.exprcode not in [gcc.EqExpr, gcc.NeExpr]:
            # The SMT solver API does not handle <, <=, >, >= for Boolean
            # values, so we convert them to bit-vectors.
            is_unsigned = True
            one = BitVecVal(1, 1)
            zero = BitVecVal(0, 1)
            rhs0 = If(rhs0, one, zero)
            rhs1 = If(rhs1, one, zero)
    elif isinstance(type, gcc.PointerType):
        is_unsigned = True
        rhs0 = rhs0.bitvector()
        rhs1 = rhs1.bitvector()
    else:
        raise NotImplementedError(f"process_comparison {type.__class__}", stmt.loc)

    if isinstance(type, gcc.RealType):
        if stmt.exprcode == gcc.LtExpr:
            return fpLT(rhs0, rhs1)
        if stmt.exprcode == gcc.LeExpr:
            return fpLEQ(rhs0, rhs1)
        if stmt.exprcode == gcc.GtExpr:
            return fpGT(rhs0, rhs1)
        if stmt.exprcode == gcc.GeExpr:
            return fpGEQ(rhs0, rhs1)
        if stmt.exprcode == gcc.EqExpr:
            return fpEQ(rhs0, rhs1)
        if stmt.exprcode == gcc.NeExpr:
            return fpNEQ(rhs0, rhs1)
        if stmt.exprcode == gcc.UneqExpr:
            return Or(Or(fpIsNaN(rhs0), fpIsNaN(rhs1)), fpEQ(rhs0, rhs1))
        if stmt.exprcode == gcc.UnltExpr:
            return Or(Or(fpIsNaN(rhs0), fpIsNaN(rhs1)), fpLT(rhs0, rhs1))
        if stmt.exprcode == gcc.UnleExpr:
            return Or(Or(fpIsNaN(rhs0), fpIsNaN(rhs1)), fpLEQ(rhs0, rhs1))
        if stmt.exprcode == gcc.UngtExpr:
            return Or(Or(fpIsNaN(rhs0), fpIsNaN(rhs1)), fpGT(rhs0, rhs1))
        if stmt.exprcode == gcc.UngeExpr:
            return Or(Or(fpIsNaN(rhs0), fpIsNaN(rhs1)), fpGEQ(rhs0, rhs1))
        if stmt.exprcode == gcc.UnorderedExpr:
            return Or(fpIsNaN(rhs0), fpIsNaN(rhs1))
        if stmt.exprcode == gcc.OrderedExpr:
            return Not(Or(fpIsNaN(rhs0), fpIsNaN(rhs1)))
        if stmt.exprcode == gcc.LtgtExpr:
            return Or(fpLT(rhs0, rhs1), fpGT(rhs0, rhs1))
    else:
        if stmt.exprcode == gcc.LtExpr:
            if is_unsigned:
                return ULT(rhs0, rhs1)
            return rhs0 < rhs1
        if stmt.exprcode == gcc.LeExpr:
            if is_unsigned:
                return ULE(rhs0, rhs1)
            return rhs0 <= rhs1
        if stmt.exprcode == gcc.GtExpr:
            if is_unsigned:
                return UGT(rhs0, rhs1)
            return rhs0 > rhs1
        if stmt.exprcode == gcc.GeExpr:
            if is_unsigned:
                return UGE(rhs0, rhs1)
            return rhs0 >= rhs1
        if stmt.exprcode == gcc.EqExpr:
            return rhs0 == rhs1
        if stmt.exprcode == gcc.NeExpr:
            return rhs0 != rhs1

    raise NotImplementedError(f"process_comparison {stmt.exprcode}", stmt.loc)


def process_integer_binary(stmt, smt_bb):
    assert len(stmt.rhs) == 2

    type = stmt.lhs.type
    rhs0 = get_tree_as_smt(stmt.rhs[0], smt_bb)
    rhs1 = get_tree_as_smt(stmt.rhs[1], smt_bb)
    if stmt.exprcode == gcc.PlusExpr:
        res = rhs0 + rhs1
        if not type.overflow_wraps:
            erhs0 = SignExt(1, rhs0)
            erhs1 = SignExt(1, rhs1)
            eres = erhs0 + erhs1
            smt_bb.add_ub(SignExt(1, res) != eres)
        return res
    if stmt.exprcode == gcc.MinusExpr:
        res = rhs0 - rhs1
        if not type.overflow_wraps:
            erhs0 = SignExt(1, rhs0)
            erhs1 = SignExt(1, rhs1)
            eres = erhs0 - erhs1
            smt_bb.add_ub(SignExt(1, res) != eres)
        return res
    if stmt.exprcode == gcc.MultExpr:
        res = rhs0 * rhs1
        if not type.overflow_wraps:
            erhs0 = SignExt(type.precision, rhs0)
            erhs1 = SignExt(type.precision, rhs1)
            eres = erhs0 * erhs1
            smt_bb.add_ub(SignExt(type.precision, res) != eres)
        return res
    if stmt.exprcode == gcc.WidenMultExpr:
        if stmt.rhs[0].type.unsigned:
            rhs0 = ZeroExt(stmt.rhs[0].type.precision, rhs0)
        else:
            rhs0 = SignExt(stmt.rhs[0].type.precision, rhs0)
        if stmt.rhs[1].type.unsigned:
            rhs1 = ZeroExt(stmt.rhs[1].type.precision, rhs1)
        else:
            rhs1 = SignExt(stmt.rhs[1].type.precision, rhs1)
        return rhs0 * rhs1
    if stmt.exprcode == gcc.MultHighpartExpr:
        precision = stmt.rhs[0].type.precision
        if stmt.rhs[0].type.unsigned:
            rhs0 = ZeroExt(precision, rhs0)
        else:
            rhs0 = SignExt(precision, rhs0)
        if stmt.rhs[1].type.unsigned:
            rhs1 = ZeroExt(precision, rhs1)
        else:
            rhs1 = SignExt(precision, rhs1)
        result = rhs0 * rhs1
        return Extract(2 * precision - 1, precision, result)
    if stmt.exprcode == gcc.TruncDivExpr:
        if type.unsigned:
            res = UDiv(rhs0, rhs1)
        else:
            res = rhs0 / rhs1
            if not type.overflow_wraps:
                ub = And(rhs0 == type.min_value.constant, rhs1 == -1)
                smt_bb.add_ub(ub)
        smt_bb.add_ub(rhs1 == 0)
        return res
    if stmt.exprcode == gcc.ExactDivExpr:
        if type.unsigned:
            res = UDiv(rhs0, rhs1)
        else:
            res = rhs0 / rhs1
            if not type.overflow_wraps:
                ub = And(rhs0 == type.min_value.constant, rhs1 == -1)
                smt_bb.add_ub(ub)
            if type.unsigned:
                smt_bb.add_ub(URem(rhs0, rhs1) != 0)
            else:
                smt_bb.add_ub(SRem(rhs0, rhs1) != 0)
        smt_bb.add_ub(rhs1 == 0)
        return res
    if stmt.exprcode == gcc.TruncModExpr:
        if type.unsigned:
            res = URem(rhs0, rhs1)
        else:
            res = SRem(rhs0, rhs1)
            if not type.overflow_wraps:
                smt_bb.add_ub(And(rhs0 == type.min_value.constant, rhs1 == -1))
        smt_bb.add_ub(rhs1 == 0)
        return res
    if stmt.exprcode == gcc.BitIorExpr:
        return rhs0 | rhs1
    if stmt.exprcode == gcc.BitAndExpr:
        return rhs0 & rhs1
    if stmt.exprcode == gcc.BitXorExpr:
        return rhs0 ^ rhs1
    if stmt.exprcode == gcc.LshiftExpr:
        smt_bb.add_ub(UGE(rhs1, type.precision))
        if stmt.rhs[0].type.precision != stmt.rhs[1].type.precision:
            rhs1 = convert_to_integer(rhs1, stmt.rhs[1].type, stmt.rhs[0].type)
        return rhs0 << rhs1
    if stmt.exprcode == gcc.RshiftExpr:
        smt_bb.add_ub(UGE(rhs1, type.precision))
        if stmt.rhs[0].type.precision != stmt.rhs[1].type.precision:
            rhs1 = convert_to_integer(rhs1, stmt.rhs[1].type, stmt.rhs[0].type)
        if type.unsigned:
            res = LShR(rhs0, rhs1)
        else:
            res = rhs0 >> rhs1
        return res
    if stmt.exprcode == gcc.MaxExpr:
        if type.unsigned:
            cmp = UGE(rhs0, rhs1)
        else:
            cmp = rhs0 >= rhs1
        return If(cmp, rhs0, rhs1)
    if stmt.exprcode == gcc.MinExpr:
        if type.unsigned:
            cmp = ULE(rhs0, rhs1)
        else:
            cmp = rhs0 <= rhs1
        return If(cmp, rhs0, rhs1)
    if stmt.exprcode == gcc.LrotateExpr:
        smt_bb.add_ub(UGE(rhs1, type.precision))
        if stmt.rhs[0].type.precision != stmt.rhs[1].type.precision:
            rhs1 = convert_to_integer(rhs1, stmt.rhs[1].type, stmt.rhs[0].type)
        return RotateLeft(rhs0, rhs1)
    if stmt.exprcode == gcc.RrotateExpr:
        smt_bb.add_ub(UGE(rhs1, type.precision))
        if stmt.rhs[0].type.precision != stmt.rhs[1].type.precision:
            rhs1 = convert_to_integer(rhs1, stmt.rhs[1].type, stmt.rhs[0].type)
        return RotateRight(rhs0, rhs1)
    if stmt.exprcode == gcc.PointerDiffExpr:
        assert type.precision == 64
        bv0 = rhs0.bitvector()
        bv1 = rhs1.bitvector()
        return bv0 - bv1

    raise NotImplementedError(f"process_integer_binary {stmt.exprcode}", stmt.loc)


def process_boolean_binary(stmt, smt_bb):
    assert len(stmt.rhs) == 2

    rhs0 = get_tree_as_smt(stmt.rhs[0], smt_bb)
    rhs1 = get_tree_as_smt(stmt.rhs[1], smt_bb)

    if stmt.exprcode in [
        gcc.LtExpr,
        gcc.LeExpr,
        gcc.GtExpr,
        gcc.GeExpr,
        gcc.EqExpr,
        gcc.NeExpr,
        gcc.UneqExpr,
        gcc.UnltExpr,
        gcc.UnleExpr,
        gcc.UngtExpr,
        gcc.UngeExpr,
        gcc.UnorderedExpr,
        gcc.OrderedExpr,
        gcc.LtgtExpr,
    ]:
        return process_comparison(stmt, smt_bb)

    rhs0 = canonicalize_bool(rhs0)
    rhs1 = canonicalize_bool(rhs1)
    if stmt.exprcode in (gcc.BitIorExpr, gcc.MaxExpr):
        return Or(rhs0, rhs1)
    if stmt.exprcode in (gcc.BitAndExpr, gcc.MinExpr):
        return And(rhs0, rhs1)
    if stmt.exprcode == gcc.BitXorExpr:
        return Xor(rhs0, rhs1)

    raise NotImplementedError(f"process_boolean_binary {stmt.exprcode}", stmt.loc)


def process_float_binary(stmt, smt_bb):
    assert len(stmt.rhs) == 2

    rhs0 = get_tree_as_smt(stmt.rhs[0], smt_bb)
    rhs1 = get_tree_as_smt(stmt.rhs[1], smt_bb)
    if stmt.exprcode == gcc.PlusExpr:
        return rhs0 + rhs1
    if stmt.exprcode == gcc.MinusExpr:
        return rhs0 - rhs1
    if stmt.exprcode == gcc.MultExpr:
        return rhs0 * rhs1
    if stmt.exprcode == gcc.RdivExpr:
        return rhs0 / rhs1

    raise NotImplementedError(f"process_float_binary {stmt.exprcode}", stmt.loc)


def process_pointer_binary(stmt, smt_bb):
    assert len(stmt.rhs) == 2

    rhs0 = get_tree_as_smt(stmt.rhs[0], smt_bb)
    rhs1 = get_tree_as_smt(stmt.rhs[1], smt_bb)
    if stmt.exprcode == gcc.PointerPlusExpr:
        assert rhs1.sort() == BitVecSort(64)
        # Note: add_to_offset adds the UB check.
        offset = add_to_offset(rhs0.offset, rhs1, smt_bb)
        return SmtPointer(rhs0.mem_id, offset)
    if stmt.exprcode == gcc.MinExpr:
        bv0 = rhs0.bitvector()
        bv1 = rhs1.bitvector()
        return build_if(bv0 < bv1, rhs0, rhs1)
    if stmt.exprcode == gcc.MaxExpr:
        bv0 = rhs0.bitvector()
        bv1 = rhs1.bitvector()
        return build_if(bv0 >= bv1, rhs0, rhs1)
    if stmt.exprcode == gcc.BitAndExpr:
        mem_id = rhs0.mem_id & rhs1.mem_id
        offset = rhs0.offset & rhs1.offset
        return SmtPointer(mem_id, offset)

    raise NotImplementedError(f"process_pointer_binary {stmt.exprcode}", stmt.loc)


def canonicalize_bool(expr):
    """Change BitVecSort(1) to BoolSort().
    Some passes treat <unnamed-unsigned:1> as _Bool, and we may end up with,
    for example, xor of a  BitVecSort(1) and BoolSort()."""
    if expr.sort() == BitVecSort(1):
        return expr != 0
    assert expr.sort() == BoolSort()
    return expr


def add_to_offset(offset, val, smt_bb):
    "Add a 64bit value to a pointer offset, and report UB on overflow."
    assert offset.sort() == BitVecSort(PTR_OFFSET_BITS)
    assert val.sort() == BitVecSort(64)
    is_constant = is_bv_value(offset)
    offset = ZeroExt(64 - PTR_OFFSET_BITS, offset)
    offset = offset + val
    if is_constant:
        offset = simplify(offset)
    smt_bb.add_ub(UGE(offset, (1 << PTR_OFFSET_BITS)))
    offset = Extract(PTR_OFFSET_BITS - 1, 0, offset)
    if is_constant:
        offset = simplify(offset)
    return offset


def bit_cast(value, src_type, dest_type):
    if isinstance(src_type, gcc.RealType):
        value = fpToIEEEBV(value)
    elif isinstance(src_type, gcc.PointerType):
        value = value.bitvector()
    elif not isinstance(src_type, gcc.IntegerType):
        raise NotImplementedError(f"bit_cast src_type {src_type.__class__}")

    if isinstance(dest_type, gcc.RealType):
        return fpBVToFP(value, get_smt_sort(dest_type))
    if isinstance(dest_type, gcc.PointerType):
        mem_id = Extract(63, PTR_OFFSET_BITS, value)
        offset = Extract(PTR_OFFSET_BITS - 1, 0, value)
        return SmtPointer(mem_id, offset)
    if isinstance(dest_type, gcc.IntegerType):
        return value

    raise NotImplementedError(f"bit_cast dest_type {dest_type.__class__}")


def convert_to_integer(value, src_type, dest_type):
    assert isinstance(dest_type, gcc.IntegerType)
    if isinstance(src_type, gcc.IntegerType):
        if src_type.precision > dest_type.precision:
            return Extract(dest_type.precision - 1, 0, value)
        if src_type.precision < dest_type.precision:
            extbits = dest_type.precision - src_type.precision
            if src_type.unsigned:
                return ZeroExt(extbits, value)
            return SignExt(extbits, value)
        return value
    if isinstance(src_type, gcc.BooleanType):
        if value.sort() == BitVecSort(1):
            return ZeroExt(dest_type.precision - 1, value)
        bv0 = BitVecVal(0, dest_type.precision)
        bv1 = BitVecVal(1, dest_type.precision)
        return If(value, bv1, bv0)
    if isinstance(src_type, gcc.PointerType):
        res = value.bitvector()
        if dest_type.precision < 64:
            res = Extract(dest_type.precision - 1, 0, res)
        elif dest_type.precision > 64:
            res = ZeroExt(dest_type.precision - 64, res)
        return res

    raise NotImplementedError(f"convert_to_integer {src_type.__class__}")


def convert_to_pointer(value, src_type, dest_type):
    assert isinstance(dest_type, gcc.PointerType)

    if isinstance(src_type, gcc.BooleanType):
        mem_id = BitVecVal(0, PTR_ID_BITS)
        bv0 = BitVecVal(0, PTR_OFFSET_BITS)
        bv1 = BitVecVal(1, PTR_OFFSET_BITS)
        offset = If(value, bv1, bv0)
        return SmtPointer(mem_id, offset)
    if isinstance(src_type, gcc.IntegerType):
        if src_type.precision < 64:
            extbits = 64 - src_type.precision
            if src_type.unsigned:
                value = ZeroExt(extbits, value)
            else:
                value = SignExt(extbits, value)
        mem_id = Extract(63, PTR_OFFSET_BITS, value)
        offset = Extract(PTR_OFFSET_BITS - 1, 0, value)
        return SmtPointer(mem_id, offset)
    if isinstance(src_type, gcc.PointerType):
        return value

    raise NotImplementedError(f"convert_to_pointer {src_type.__class__}")


def convert_to_boolean(value, src_type):
    if isinstance(src_type, gcc.IntegerType):
        return (value & 1) != 0

    raise NotImplementedError(f"convert_to_boolean {src_type.__class__}")


def convert_to_float(value, src_type, dest_type):
    assert isinstance(dest_type, gcc.RealType)

    smt_sort = get_smt_sort(dest_type)
    if isinstance(src_type, gcc.RealType):
        return fpToFP(RNE(), value, smt_sort)
    if isinstance(src_type, gcc.BooleanType):
        fp0 = FPVal(0.0, smt_sort)
        fp1 = FPVal(1.0, smt_sort)
        return If(value, fp1, fp0)
    if isinstance(src_type, gcc.IntegerType):
        return fpToFP(RNE(), value, smt_sort)

    raise NotImplementedError(f"convert_to_float {src_type.__class__}")


def process_unary(stmt, smt_bb, uninit_is_ub=True):
    assert len(stmt.rhs) == 1

    if isinstance(
        stmt.rhs[0],
        (gcc.VarDecl, gcc.ArrayRef, gcc.ComponentRef, gcc.BitFieldRef, gcc.MemRef),
    ):
        res, is_initialized = load(stmt, smt_bb)
        if isinstance(stmt.lhs, gcc.SsaName):
            smt_bb.smt_fun.tree_is_initialized[stmt.lhs] = is_initialized
        return res

    # We want to treat load of an uninitialized local value in the same
    # way as read of an uninitialized global value. But uninitialized
    # local values are encoded as gcc.SsaName, so we need to disable the
    # check for undefined value in get_tree_as_smt and do the check here
    # when we know if the value is just placed in a new gcc.SsaName (which
    # is allowed) or used in an operation (which is UB).
    rhs = get_tree_as_smt(stmt.rhs[0], smt_bb, False)
    if stmt.exprcode == gcc.SsaName:
        if stmt.rhs[0] in smt_bb.smt_fun.tree_is_initialized:
            is_initialized = smt_bb.smt_fun.tree_is_initialized[stmt.rhs[0]]
            smt_bb.smt_fun.tree_is_initialized[stmt.lhs] = is_initialized
        return rhs

    if uninit_is_ub and stmt.rhs[0] in smt_bb.smt_fun.tree_is_initialized:
        is_initialized = smt_bb.smt_fun.tree_is_initialized[stmt.rhs[0]]
        if len(is_initialized) == 1:
            is_uninitialized = Not(is_initialized[0])
        else:
            is_uninitialized = Not(And(is_initialized))
        smt_bb.add_ub(is_uninitialized)

    if stmt.exprcode == gcc.NopExpr:
        src_type = stmt.rhs[0].type
        dest_type = stmt.lhs.type
        if isinstance(dest_type, gcc.IntegerType):
            return convert_to_integer(rhs, src_type, dest_type)
        if isinstance(dest_type, gcc.PointerType):
            return convert_to_pointer(rhs, src_type, dest_type)
        if isinstance(dest_type, gcc.BooleanType):
            return convert_to_boolean(rhs, src_type)
        if isinstance(dest_type, gcc.RealType):
            return convert_to_float(rhs, src_type, dest_type)
    elif stmt.exprcode == gcc.FloatExpr:
        return convert_to_float(rhs, stmt.rhs[0].type, stmt.lhs.type)
    elif stmt.exprcode == gcc.FixTruncExpr:
        assert isinstance(stmt.lhs.type, gcc.IntegerType)
        smt_bb.add_ub(Or(fpIsInf(rhs), fpIsNaN(rhs)))
        val = fpRoundToIntegral(RTZ(), rhs)
        smt_sort = get_smt_sort(stmt.lhs.type)
        precision = stmt.lhs.type.precision
        min_val = stmt.lhs.type.min_value.constant
        max_val = stmt.lhs.type.max_value.constant
        if stmt.lhs.type.unsigned:
            max_as_float = fpUnsignedToFP(
                RTZ(), BitVecVal(max_val, precision), rhs.sort()
            )
            min_as_float = fpUnsignedToFP(
                RTZ(), BitVecVal(min_val, precision), rhs.sort()
            )
            smt_bb.add_ub(Or(val < min_as_float, val > max_as_float))
            return fpToUBV(RTZ(), val, smt_sort)
        max_as_float = fpSignedToFP(RTZ(), BitVecVal(max_val, precision), rhs.sort())
        min_as_float = fpSignedToFP(RTZ(), BitVecVal(min_val, precision), rhs.sort())
        smt_bb.add_ub(Or(val < min_as_float, val > max_as_float))
        return fpToSBV(RTZ(), val, smt_sort)
    elif stmt.exprcode == gcc.NegateExpr:
        type = stmt.rhs[0].type
        if isinstance(type, gcc.IntegerType) and not type.overflow_wraps:
            min_int = 1 << (type.precision - 1)
            smt_bb.add_ub(rhs == min_int)
        return -rhs
    elif stmt.exprcode == gcc.BitNotExpr:
        if isinstance(stmt.lhs.type, gcc.BooleanType):
            return Not(rhs)
        if isinstance(stmt.lhs.type, gcc.IntegerType):
            return ~rhs
    elif stmt.exprcode == gcc.AbsExpr:
        if isinstance(stmt.lhs.type, gcc.RealType):
            return fpAbs(rhs)
        if isinstance(stmt.lhs.type, gcc.IntegerType):
            assert not stmt.lhs.type.unsigned
            if not stmt.lhs.type.overflow_wraps:
                min_int = 1 << (stmt.lhs.type.precision - 1)
                smt_bb.add_ub(rhs == min_int)
            return If(rhs >= 0, rhs, -rhs)
    elif hasattr(gcc, "AbsuExpr") and stmt.exprcode == gcc.AbsuExpr:
        if isinstance(stmt.lhs.type, gcc.IntegerType):
            return If(rhs >= 0, rhs, -rhs)
    elif stmt.exprcode == gcc.ViewConvertExpr:
        # The actual work is done in get_tree_as_smt.
        return rhs
    elif stmt.exprcode == gcc.ParenExpr:
        return rhs
    elif stmt.exprcode in [
        gcc.ParmDecl,
        gcc.IntegerCst,
        gcc.RealCst,
        gcc.AddrExpr,
    ]:
        return rhs

    raise NotImplementedError(
        f"process_unary {stmt.exprcode} {stmt.lhs.type.__class__}", stmt.loc
    )


def process_ternary(stmt, smt_bb):
    assert len(stmt.rhs) == 3

    rhs0 = get_tree_as_smt(stmt.rhs[0], smt_bb)
    rhs1 = get_tree_as_smt(stmt.rhs[1], smt_bb)
    rhs2 = get_tree_as_smt(stmt.rhs[2], smt_bb)
    if stmt.exprcode == gcc.CondExpr:
        return build_if(rhs0, rhs1, rhs2)

    raise NotImplementedError(f"process_ternary {stmt.exprcode}", stmt.loc)


def build_full_edge_cond(edge, smt_fun):
    "Return a SMT condition for when the destination is executed."
    src_smt_bb = smt_fun.bb2smt[edge.src]
    if len(edge.src.succs) == 1:
        cond = src_smt_bb.is_executed
    elif src_smt_bb.switch_label_to_cond is not None:
        dest_smt_bb = smt_fun.bb2smt[edge.dest]
        cond = None
        for label in dest_smt_bb.labels:
            if label not in src_smt_bb.switch_label_to_cond:
                # The destination BB may have additional labels not
                # used by the switch statement. Ignore them.
                continue
            label_cond = src_smt_bb.switch_label_to_cond[label]
            if label_cond is not None:
                if cond is None:
                    cond = label_cond
                else:
                    cond = Or(cond, label_cond)
        assert cond is not None
        if not is_true(src_smt_bb.is_executed):
            cond = And(cond, src_smt_bb.is_executed)
    else:
        assert len(edge.src.succs) == 2
        if edge.true_value:
            cond = src_smt_bb.smtcond
        else:
            cond = Not(src_smt_bb.smtcond)
        if not is_true(src_smt_bb.is_executed):
            cond = And(cond, src_smt_bb.is_executed)
    return cond


def build_if(cond, val1, val2):
    res = val1
    if isinstance(val1, SmtPointer):
        if not (val1.mem_id == val2.mem_id and val1.offset == val2.offset):
            if val1.mem_id == val2.mem_id:
                mem_id = val1.mem_id
            else:
                mem_id = If(cond, val1.mem_id, val2.mem_id)
            if val1.offset == val2.offset:
                offset = val1.offset
            else:
                offset = If(cond, val1.offset, val2.offset)
            res = SmtPointer(mem_id, offset)
    else:
        if not val1 == val2:
            res = If(cond, val1, val2)
    return res


def process_phi_smt_args(args, smt_fun):
    "Build SMT for a list of phi SMT args (i.e. a list of (smt_val, edge))"
    res, _ = args[0]
    for value, edge in args[1:]:
        cond = build_full_edge_cond(edge, smt_fun)
        res = build_if(cond, value, res)
    return res


def process_phi(args, smt_bb, lhs=None):
    smt_args = []
    is_init = []
    need_uninit_check = False
    type = args[0][0].type
    if isinstance(type, gcc.IntegerType) and type.sizeof * 8 != type.precision:
        sizeof = (type.precision + 7) // 8
    else:
        sizeof = type.sizeof
    for expr, edge in args:
        value = get_tree_as_smt(expr, smt_bb, False)
        if expr in smt_bb.smt_fun.tree_is_initialized:
            is_initialized = smt_bb.smt_fun.tree_is_initialized[expr]
            need_uninit_check = True
        else:
            is_initialized = [BoolVal(True)] * sizeof
        smt_args.append((value, edge))
        is_init.append((is_initialized, edge))
    if need_uninit_check and lhs is not None:
        is_initialized = []
        for i in range(0, sizeof):
            is_init2 = []
            for initialized, edge in is_init:
                is_init2.append((initialized[i], edge))
            is_initialized.append(process_phi_smt_args(is_init2, smt_bb.smt_fun))
        smt_bb.smt_fun.tree_is_initialized[lhs] = is_initialized
    return process_phi_smt_args(smt_args, smt_bb.smt_fun)


def process_GimpleCall(stmt, smt_bb):
    if stmt.fndecl is None:
        raise NotImplementedError("GimpleCall None", stmt.loc)
    if stmt.fndecl.name == "__builtin_unreachable":
        smt_bb.add_ub(BoolVal(True))
        return
    if stmt.fndecl.name == "__builtin_assume_aligned":
        ptr = get_tree_as_smt(stmt.rhs[2], smt_bb)
        alignment = stmt.rhs[3].constant
        if alignment > 1:
            smt_bb.add_ub((ptr.offset & (alignment - 1)) != 0)
        smt_bb.smt_fun.tree_to_smt[stmt.lhs] = ptr
        return
    if stmt.fndecl.name == "__builtin_memset":
        ptr = get_tree_as_smt(stmt.rhs[2], smt_bb)
        val = get_tree_as_smt(stmt.rhs[3], smt_bb)
        size = simplify(get_tree_as_smt(stmt.rhs[4], smt_bb))
        if is_bv_value(size):
            size = int(str(size))
            if size < MAX_MEMORY_UNROLL_LIMIT:
                val = Extract(7, 0, val)
                is_initialized = [BoolVal(True)]
                for i in range(0, size):
                    store_bytes(
                        ptr.mem_id, ptr.offset + i, 1, val, is_initialized, smt_bb
                    )
                return
    if stmt.fndecl.name in [
        "__builtin_popcount",
        "__builtin_popcountl",
        "__builtin_popcountll",
    ]:
        assert isinstance(stmt.lhs, gcc.SsaName)
        value = get_tree_as_smt(stmt.rhs[2], smt_bb)
        result = BitVecVal(0, stmt.lhs.type.precision)
        for i in range(0, stmt.rhs[2].type.precision):
            bit = Extract(i, i, value)
            result = result + ZeroExt(stmt.lhs.type.precision - 1, bit)
        smt_bb.smt_fun.tree_to_smt[stmt.lhs] = result
        return
    if stmt.fndecl.name in [
        "__builtin_parity",
        "__builtin_parityl",
        "__builtin_parityll",
    ]:
        assert isinstance(stmt.lhs, gcc.SsaName)
        value = get_tree_as_smt(stmt.rhs[2], smt_bb)
        parity = BitVecVal(0, 1)
        for i in range(0, stmt.rhs[2].type.precision):
            parity = parity ^ Extract(i, i, value)
        result = ZeroExt(stmt.lhs.type.precision - 1, parity)
        smt_bb.smt_fun.tree_to_smt[stmt.lhs] = result
        return
    if stmt.fndecl.name in [
        "__builtin_copysignf",
        "__builtin_copysign",
        "__builtin_copysignl",
    ]:
        assert isinstance(stmt.lhs, gcc.SsaName)
        x = get_tree_as_smt(stmt.rhs[2], smt_bb)
        y = get_tree_as_smt(stmt.rhs[3], smt_bb)
        precision = stmt.rhs[2].type.precision
        x_bits = fpToIEEEBV(x)
        y_bits = fpToIEEEBV(y)
        y_signbit = Extract(precision - 1, precision - 1, y_bits)
        result_bits = Concat([y_signbit, Extract(precision - 2, 0, x_bits)])
        smt_bb.smt_fun.tree_to_smt[stmt.lhs] = fpBVToFP(result_bits, y.sort())
        return
    if stmt.fndecl.name in [
        "__builtin_bswap16",
        "__builtin_bswap32",
        "__builtin_bswap64",
        "__builtin_bswap128",
    ]:
        value = get_tree_as_smt(stmt.rhs[2], smt_bb)
        bytes = []
        for i in range(0, stmt.lhs.type.precision // 8):
            bytes.append(Extract(i * 8 + 7, i * 8, value))
        assert isinstance(stmt.lhs, gcc.SsaName)
        smt_bb.smt_fun.tree_to_smt[stmt.lhs] = Concat(bytes)
        return

    name = ""
    if stmt.fndecl.name.startswith("__builtin"):
        # We skip showing non-builtin names as they only adds noise in
        # the logs. But showing which builtins are called is useful as
        # we may want to implement support for some of them.
        name = stmt.fndecl.name
    raise NotImplementedError(f"GimpleCall {name}", stmt.loc)


def process_ArrayRef(array_ref, smt_bb):
    assert isinstance(array_ref, gcc.ArrayRef)
    elem_size = array_ref.type.sizeof
    if isinstance(array_ref.index, gcc.IntegerCst):
        offset = elem_size * array_ref.index.constant
        if array_ref.array.type.range is None:
            if offset > (1 << PTR_OFFSET_BITS) or offset < 0:
                smt_bb.add_ub(BoolVal(True))
        else:
            # TODO: Handle "one past" when not dereferenced.
            range = array_ref.array.type.range
            assert range.min_value.constant == 0
            if array_ref.index.constant > range.max_value.constant:
                smt_bb.add_ub(BoolVal(True))
        return array_ref.array, BitVecVal(offset, 64)
    if isinstance(array_ref.index, gcc.SsaName):
        index = get_tree_as_smt(array_ref.index, smt_bb)
        precision = array_ref.index.type.precision
        if precision < 64:
            if array_ref.index.type.unsigned:
                index = ZeroExt(64 - precision, index)
            else:
                index = SignExt(64 - precision, index)
        elif precision > 64:
            smt_bb.add_ub(UGE(index, (1 << 64)))
            index = Extract(63, 0, index)
        offset = BitVecVal(elem_size, 64) * index
        if array_ref.array.type.range is None:
            eindex = ZeroExt(64, index)
            eoffset = BitVecVal(elem_size, 128) * eindex
            smt_bb.add_ub(UGE(eoffset, (1 << PTR_OFFSET_BITS)))
        else:
            # TODO: Handle "one past" when not dereferenced.
            range = array_ref.array.type.range
            assert range.min_value.constant == 0
            smt_bb.add_ub(UGT(index, range.max_value.constant))
        return array_ref.array, offset
    raise NotImplementedError(
        f"process_ArrayRef {array_ref.index.__class__}", array_ref.location
    )


def build_smt_addr(expr, smt_bb):
    "Return a mem_id/offset pair for an expr representing an address."
    if isinstance(expr, gcc.MemRef):
        ptr = get_tree_as_smt(expr.operand, smt_bb)
        assert isinstance(expr.offset, gcc.IntegerCst)
        assert isinstance(expr.offset.type, gcc.PointerType)
        mem_ref_offset = BitVecVal(expr.offset.constant, 64)
        offset = add_to_offset(ptr.offset, mem_ref_offset, smt_bb)
        if expr.type.alignmentof > 1:
            smt_bb.add_ub((offset & (expr.type.alignmentof - 1)) != 0)
        return ptr.mem_id, offset
    if isinstance(expr, gcc.VarDecl):
        assert expr in smt_bb.smt_fun.decl_to_id
        mem_id = smt_bb.smt_fun.decl_to_id[expr]
        return mem_id, BitVecVal(0, PTR_OFFSET_BITS)

    if isinstance(expr, gcc.ArrayRef):
        decl, offset = process_ArrayRef(expr, smt_bb)
    elif isinstance(expr, gcc.ComponentRef):
        decl = expr.target
        offset = BitVecVal(expr.offset, 64)
    elif isinstance(expr, gcc.BitFieldRef):
        if expr.position.constant % 8 != 0:
            raise NotImplementedError(f"build_smt_addr {expr.__class__}")
        decl = expr.operand
        offset = BitVecVal(expr.position.constant // 8, 64)
    else:
        raise NotImplementedError(f"build_smt_addr {expr.__class__}")
    mem_id, offset2 = build_smt_addr(decl, smt_bb)
    new_offset = add_to_offset(offset2, offset, smt_bb)
    return mem_id, new_offset


def process_GimpleAssign(stmt, smt_bb):
    if not isinstance(stmt.lhs, gcc.SsaName):
        store(stmt, smt_bb)
        return

    if len(stmt.rhs) == 1:
        rhs = process_unary(stmt, smt_bb)
    elif len(stmt.rhs) == 3:
        rhs = process_ternary(stmt, smt_bb)
    elif isinstance(stmt.lhs.type, gcc.IntegerType):
        rhs = process_integer_binary(stmt, smt_bb)
    elif isinstance(stmt.lhs.type, gcc.RealType):
        rhs = process_float_binary(stmt, smt_bb)
    elif isinstance(stmt.lhs.type, gcc.BooleanType):
        rhs = process_boolean_binary(stmt, smt_bb)
    elif isinstance(stmt.lhs.type, gcc.PointerType):
        rhs = process_pointer_binary(stmt, smt_bb)
    else:
        raise NotImplementedError(
            f"process_GimpleAssign {stmt.exprcode} {stmt.lhs.type.__class__}", stmt.loc
        )
    smt_bb.smt_fun.tree_to_smt[stmt.lhs] = rhs


def process_bb(bb, smt_fun):
    smt_bb = smt_fun.bb2smt[bb]
    for phi in bb.phi_nodes:
        if isinstance(phi.lhs.type, gcc.VoidType):
            # Skip phi nodes for the memory SSA virtual SSA names.
            continue
        smt_fun.tree_to_smt[phi.lhs] = process_phi(phi.args, smt_bb, phi.lhs)

    for stmt in bb.gimple:
        if isinstance(stmt, gcc.GimpleAssign):
            process_GimpleAssign(stmt, smt_bb)
        elif isinstance(stmt, gcc.GimpleReturn):
            # stmt.retval may be None for paths where the function returns
            # without providing a value (this is allowed in C as long
            # as the caller does not use the returned value).
            if stmt.retval is not None:
                if isinstance(
                    stmt.retval,
                    (
                        gcc.VarDecl,
                        gcc.ArrayRef,
                        gcc.ComponentRef,
                        gcc.BitFieldRef,
                        gcc.MemRef,
                    ),
                ):
                    retval, is_initialized = load(stmt, smt_bb)
                    if len(is_initialized) == 1:
                        is_uninitialized = Not(is_initialized[0])
                    else:
                        is_uninitialized = Not(And(is_initialized))
                    smt_bb.add_ub(is_uninitialized)
                else:
                    retval = get_tree_as_smt(stmt.retval, smt_bb)
                if isinstance(retval, SmtPointer):
                    retval = retval.bitvector()
                smt_bb.retval = retval
        elif isinstance(stmt, gcc.GimpleCond):
            assert smt_bb.smtcond is None
            cond = process_comparison(stmt, smt_bb)
            if bb.succs[0].true_value:
                true_bb = bb.succs[0].dest
                false_bb = bb.succs[1].dest
            else:
                true_bb = bb.succs[1].dest
                false_bb = bb.succs[0].dest
            smt_bb.smtcond = cond
            smt_bb.true_bb = true_bb
            smt_bb.false_bb = false_bb
        elif isinstance(stmt, gcc.GimpleSwitch):
            assert smt_bb.switch_label_to_cond is None
            smt_bb.switch_label_to_cond = build_switch_label_to_cond(stmt, smt_bb)
        elif isinstance(stmt, gcc.GimpleLabel):
            # The label has already been handled when SmtBB was created.
            pass
        elif isinstance(stmt, gcc.GimpleCall):
            process_GimpleCall(stmt, smt_bb)
        elif isinstance(stmt, gcc.GimplePredict):
            pass
        elif isinstance(stmt, gcc.GimpleNop):
            # GimpleNop does not do anything. Ignore it.
            pass
        else:
            raise NotImplementedError(f"process_bb {stmt.__class__}", stmt.loc)


def is_const_type(type):
    if isinstance(type, (gcc.PointerType, gcc.UnionType)):
        # gcc.PointerType etc. does not have any .const attribute, so assume
        # it is not a constant.
        # TODO: Add support to the Python plugin.
        return False
    if isinstance(type, gcc.ArrayType):
        # gcc.ArrayType does not have a const qualifier, so we need to
        # check the element type (and recursively so that we handle
        # arrays of arrays.
        return is_const_type(type.dereference)
    return type.const


def is_const(expr):
    # There are two ways to say that an array is constant:
    # * Make the elements of a const type (this is how the C frontend does it).
    # * Set the TREE_CONSTANT flag on the decl (this is how the switchconv
    #   pass does it for the CSWTCH arrays it creates).
    return expr.is_constant or is_const_type(expr.type)


def init_bytes(mem_id, offset, size, value, memory, is_initialized):
    ptr = Concat([mem_id, offset])
    if is_bv_value(mem_id) and is_bv_value(offset):
        ptr = simplify(ptr)
    for i in range(0, size):
        byte_value = Extract(i * 8 + 7, i * 8, value)
        if is_bv_value(value):
            byte_value = simplify(byte_value)
        p = ptr + i
        if is_bv_value(ptr):
            p = simplify(p)
        memory = Store(memory, p, byte_value)
        is_initialized = Store(is_initialized, p, BoolVal(True))
    return memory, is_initialized


def init_global_var_decl(decl, mem_id, size, memory, is_initialized):
    assert isinstance(decl, gcc.VarDecl)

    # decl.initial may not initialize all elements, and the remaining
    # must be initialized by 0. So we start by initializing all to 0.
    # TODO: Should only do this for the bytes that are not initialized by
    # decl.initial.
    value = BitVecVal(0, decl.type.sizeof * 8)
    offset = BitVecVal(0, PTR_OFFSET_BITS)
    memory, is_initialized = init_bytes(
        mem_id, offset, size, value, memory, is_initialized
    )
    if decl.initial is None:
        return memory, is_initialized

    if isinstance(decl.initial.type, gcc.IntegerType):
        assert isinstance(decl.initial, gcc.IntegerCst)
        assert size == decl.initial.type.precision // 8
        value = BitVecVal(decl.initial.constant, decl.initial.type.precision)
        offset = BitVecVal(0, PTR_OFFSET_BITS)
        return init_bytes(mem_id, offset, size, value, memory, is_initialized)
    if isinstance(decl.initial.type, gcc.RealType):
        assert isinstance(decl.initial, gcc.RealCst)
        value = fpToIEEEBV(
            FPVal(decl.initial.constant, get_smt_sort(decl.initial.type))
        )
        if decl.initial.type.precision == 80:
            value = Concat([BitVecVal(0, 49), value])
        offset = BitVecVal(0, PTR_OFFSET_BITS)
        return init_bytes(mem_id, offset, size, value, memory, is_initialized)
    if isinstance(decl.initial, gcc.StringCst):
        bytes = []
        for c in decl.initial.constant:
            bytes.insert(0, BitVecVal(ord(c), 8))
        if len(bytes) < size:
            bytes = [BitVecVal(0, 8)] * (size - len(bytes)) + bytes
        assert len(bytes) == size
        if size == 1:
            value == bytes[0]
        else:
            value = Concat(bytes)
        offset = BitVecVal(0, PTR_OFFSET_BITS)
        return init_bytes(mem_id, offset, size, value, memory, is_initialized)
    if isinstance(decl.initial.type, (gcc.ArrayType, gcc.RecordType)):
        if not isinstance(decl.initial, gcc.Constructor):
            raise NotImplementedError(
                f"init_global_var_decl {decl.initial.__class__} instead of gcc.Constructor"
            )
        if decl.initial.is_clobber:
            is_initialized = mark_mem_uninitialized(
                mem_id, offset, size, is_initialized
            )
        if not decl.initial.no_clearing:
            # Note: It is enough to only update memory as is_initialized
            # already is true per default for global memory.
            ptr = simplify(Concat([mem_id, BitVecVal(0, PTR_OFFSET_BITS)]))
            for i in range(0, size):
                p = simplify(ptr + i)
                memory = Store(memory, p, BitVecVal(0, 8))
        for elem in decl.initial.elements:
            if isinstance(decl.initial.type, gcc.ArrayType):
                assert isinstance(elem[0], gcc.IntegerCst)
                elem_size = decl.initial.type.dereference.sizeof
                index = elem[0].constant
                offset = BitVecVal(index * elem_size, PTR_OFFSET_BITS)
            else:
                assert isinstance(elem[0], gcc.FieldDecl)
                if not isinstance(elem[0].type, gcc.IntegerType):
                    raise NotImplementedError(
                        f"init_global_var_decl {elem[0].type.__class__}"
                    )
                if elem[0].bitoffset % 8 != 0 or elem[0].type.precision % 8 != 0:
                    raise NotImplementedError(f"init_global_var_decl bitfield")
                offset = BitVecVal(elem[0].offset, PTR_OFFSET_BITS)
            if isinstance(elem[1].type, gcc.IntegerType):
                value = BitVecVal(elem[1].constant, elem[1].type.precision)
            elif isinstance(elem[1].type, gcc.RealType):
                value = fpToIEEEBV(FPVal(elem[1].constant, get_smt_sort(elem[1].type)))
                if elem[1].type.precision == 80:
                    value = Concat([BitVecVal(0, 49), value])
            else:
                raise NotImplementedError(
                    f"init_global_var_decl {elem[1].type.__class__}"
                )
            size = elem[1].type.sizeof
            memory, is_initialized = init_bytes(
                mem_id, offset, size, value, memory, is_initialized
            )
        return memory, is_initialized
    raise NotImplementedError(f"init_global_var_decl {decl.initial.type.__class__}")


def init_common_state(fun):
    memory = Array(".memory", BitVecSort(64), BitVecSort(8))
    # We must treat arbitrarily global memory as initialized (as we
    # cannot see if/what other functions has written), so let "true"
    # be the default value.
    is_initialized = K(BitVecSort(64), BoolVal(True))

    next_id = 1
    memory_objects = []
    mem_sizes = K(BitVecSort(PTR_ID_BITS), BitVecVal(0, PTR_OFFSET_BITS))
    const_mem_ids = []
    decl_to_memory = {}
    for var in gcc.get_variables():
        if isinstance(var.decl.type, gcc.ArrayType) and not isinstance(
            var.decl.type.range, gcc.IntegerType
        ):
            # This is an array declared without size. Invent a size...
            size = ANON_MEM_SIZE
        else:
            size = var.decl.type.sizeof
        memory_object = MemoryBlock(var.decl, size, next_id)
        memory_objects.append(memory_object)
        decl_to_memory[var.decl] = memory_object

        mem_id = BitVecVal(next_id, PTR_ID_BITS)
        mem_sizes = Store(mem_sizes, mem_id, BitVecVal(size, PTR_OFFSET_BITS))

        # We should not initialize non-const global memory -- it can be
        # modified before the functions are called, so functions should
        # work with any value.
        if is_const(var.decl):
            const_mem_ids.append(mem_id)
            memory, is_initialized = init_global_var_decl(
                var.decl, mem_id, size, memory, is_initialized
            )

        next_id = next_id + 1

    fun_args = []
    arg_ptrs = []
    for arg in fun.decl.arguments:
        if isinstance(arg.type, gcc.PointerType):
            ptr = Const(arg.name, BitVecSort(64))
            mem_id = Extract(63, PTR_OFFSET_BITS, ptr)
            offset = Extract(PTR_OFFSET_BITS - 1, 0, ptr)
            smt_arg = SmtPointer(mem_id, offset)
            arg_ptrs.append((smt_arg, arg.type))

            memory_object = MemoryBlock(None, ANON_MEM_SIZE, next_id)
            memory_objects.append(memory_object)

            mem_id = BitVecVal(next_id, PTR_ID_BITS)
            mem_sizes = Store(
                mem_sizes, mem_id, BitVecVal(ANON_MEM_SIZE, PTR_OFFSET_BITS)
            )

            next_id = next_id + 1
        else:
            smt_sort = get_smt_sort(arg.type)
            smt_arg = Const(arg.name, smt_sort)
        fun_args.append((smt_arg, arg.type))

    ptr_constraints = []
    for ptr, type in arg_ptrs:
        ptr_constraints.append(And(ptr.mem_id > 0, ptr.mem_id < next_id))
        smt_size = Select(mem_sizes, ptr.mem_id)
        ptr_constraints.append(And(ptr.offset >= 0, ptr.offset < smt_size))
        ptr_constraints.append(ptr.offset + type.dereference.sizeof < smt_size)
        if type.dereference.alignmentof > 1:
            ptr_constraints.append(
                (ptr.offset & (type.dereference.alignmentof - 1)) == 0
            )

    return CommonState(
        memory_objects,
        decl_to_memory,
        const_mem_ids,
        next_id,
        ptr_constraints,
        memory,
        mem_sizes,
        is_initialized,
        fun_args,
    )


def find_unimplemented(fun):
    """Check if the function contains unimplemented statements.

    This function is useful to run before doing any other work as
    it takes a long time to generate SMT for large functions, and we
    may spend minutes before finding the first unimplemented stmt."""
    for bb in fun.cfg.inverted_post_order:
        for stmt in bb.gimple:
            if isinstance(stmt, gcc.GimpleCall):
                if stmt.fndecl is None:
                    raise NotImplementedError("GimpleCall", stmt.loc)
                if not stmt.fndecl.name.startswith("__builtin"):
                    raise NotImplementedError("GimpleCall", stmt.loc)


def mark_mem_uninitialized(mem_id, offset, size, is_initialized):
    if size > MAX_MEMORY_UNROLL_LIMIT:
        raise NotImplementedError(f"mark_mem_uninitialized too big ({size})")
    ptr = Concat([mem_id, offset])
    if is_bv_value(mem_id) and is_bv_value(offset):
        ptr = simplify(ptr)
    for i in range(0, size):
        p = ptr + i
        if is_bv_value(ptr):
            p = simplify(p)
        is_initialized = Store(is_initialized, p, BoolVal(False))
    return is_initialized


def process_function(fun, state, reuse):
    if has_loop(fun):
        raise NotImplementedError("Loops", fun.decl.location)

    # Adding SMT statements is very slow in large functions. I have not
    # investigated why, but it feels like the SMT library is doing some
    # kind of garbage collection (e.g. most operations are fast, but some
    # "randomly" takes a lot of time.)
    # Most large functions cannot be checked anyway as they contain
    # function calls, etc. So we check for obviously unimplemented
    # operations before starting processing the function. It saves
    # > 2 hours when checking gcc.dg/torture/arm-fp16-int-convert-alt.c
    find_unimplemented(fun)

    smt_fun = SmtFunction(fun, state)

    decl = fun.decl
    if len(decl.arguments) != len(state.fun_args):
        raise Error("Incorrect number of arguments", fun.decl.location)
    if decl.arguments:
        for arg, smt_arg in zip(decl.arguments, state.fun_args):
            if arg.type != smt_arg[1]:
                raise Error("Incorrect type for argument", arg.location)
            smt_fun.tree_to_smt[arg] = smt_arg[0]

    mem_sizes = state.mem_sizes
    next_id = state.next_id
    memory_objects = state.global_memory[:]
    for obj in memory_objects:
        smt_fun.decl_to_id[obj.decl] = BitVecVal(obj.mem_id, PTR_ID_BITS)
    if reuse:
        for memory_object in state.local_memory:
            memory_objects.append(memory_object)
            size = memory_object.size
            mem_id = BitVecVal(memory_object.mem_id, PTR_ID_BITS)
            mem_sizes = Store(mem_sizes, mem_id, BitVecVal(size, PTR_OFFSET_BITS))

    # Some passes (such as switchconv) may add new global variables, so we
    # need to add global variables not present in the common state.
    for var in gcc.get_variables():
        if var.decl not in smt_fun.decl_to_id:
            if isinstance(var.decl.type, gcc.ArrayType) and not isinstance(
                var.decl.type.range, gcc.IntegerType
            ):
                # This is an array declared without size. Invent a size...
                size = ANON_MEM_SIZE
            else:
                size = var.decl.type.sizeof
            memory_object = MemoryBlock(var.decl, size, next_id)
            memory_objects.append(memory_object)
            mem_id = BitVecVal(next_id, PTR_ID_BITS)
            mem_sizes = Store(mem_sizes, mem_id, BitVecVal(size, PTR_OFFSET_BITS))
            # We should not initialize non-const memory -- it can be modified
            # before the functions are called, so functions should work with
            # any value.
            if is_const(var.decl):
                state.const_mem_ids.append(mem_id)
                if var.decl.initial is not None:
                    state.memory, state.is_initialized = init_global_var_decl(
                        var.decl, mem_id, size, state.memory, state.is_initialized
                    )
            next_id = next_id + 1

    local_memory = []
    for decl in smt_fun.fun.local_decls:
        if decl.static:
            assert decl in state.decl_to_memory
            # Local static decls are included in the global decls, so their
            # memory objects have already been created. But only constant
            # decls are initialized for globals, but we need to initialize
            # all local static decls because the compiler has knowledge of
            # how they are changed (unless a pointer escapes), so it can
            # optimize based on the possible values (which we do not track).
            if not is_const(decl):
                memory_object = state.decl_to_memory[decl]
                mem_id = BitVecVal(memory_object.mem_id, PTR_ID_BITS)
                size = memory_object.size
                state.memory, state.is_initialized = init_global_var_decl(
                    decl, mem_id, size, state.memory, state.is_initialized
                )
            continue

        if reuse and decl in state.decl_to_memory:
            memory_object = state.decl_to_memory[decl]
            size = memory_object.size
            mem_id = BitVecVal(memory_object.mem_id, PTR_ID_BITS)
        else:
            size = decl.type.sizeof
            memory_object = MemoryBlock(decl, size, next_id)
            local_memory.append(memory_object)
            state.decl_to_memory[decl] = memory_object
            next_id = next_id + 1
            memory_objects.append(memory_object)
            mem_id = BitVecVal(memory_object.mem_id, PTR_ID_BITS)
            mem_sizes = Store(mem_sizes, mem_id, BitVecVal(size, PTR_OFFSET_BITS))
        offset = BitVecVal(0, PTR_OFFSET_BITS)
        state.is_initialized = mark_mem_uninitialized(
            mem_id, offset, size, state.is_initialized
        )

    state.next_id = next_id
    if not reuse:
        state.local_memory = local_memory

    for obj in memory_objects:
        smt_fun.decl_to_id[obj.decl] = BitVecVal(obj.mem_id, PTR_ID_BITS)

    for bb in fun.cfg.inverted_post_order:
        SmtBB(bb, smt_fun, mem_sizes)
        process_bb(bb, smt_fun)

    if not isinstance(fun.decl.result.type, gcc.VoidType):
        results = []
        for edge in fun.cfg.exit.preds:
            src_smt_bb = smt_fun.bb2smt[edge.src]
            if src_smt_bb.retval is None:
                # The function is not returning a value when following this
                # edge. This is allowed -- it is only UB in C in the caller
                # if it is using the nonexisting value.
                # Create a symbolic constant to make checking work.
                if state.retval is None:
                    retval_sort = get_smt_sort(fun.decl.result.type)
                    state.retval = Const(".retval", retval_sort)
                retval = state.retval
            else:
                retval = src_smt_bb.retval
            results.append((retval, edge))
        if results:
            smt_fun.retval = process_phi_smt_args(results, smt_fun)
    exit_smt_bb = smt_fun.bb2smt[fun.cfg.exit]
    smt_fun.memory = exit_smt_bb.memory
    smt_fun.mem_sizes = exit_smt_bb.mem_sizes
    smt_fun.is_initialized = exit_smt_bb.is_initialized

    for bb in fun.cfg.inverted_post_order:
        smt_bb = smt_fun.bb2smt[bb]
        if smt_bb.invokes_ub is not None:
            smt_fun.add_ub(And(smt_bb.is_executed, smt_bb.invokes_ub))

    return smt_fun


def init_solver(src_smt_fun, append_ub_check=True):
    solver = Solver()
    if SOLVER_TIMEOUT > 0:
        solver.set("timeout", SOLVER_TIMEOUT * 1000)
    if SOLVER_MAX_MEMORY > 0:
        solver.set("max_memory", SOLVER_MAX_MEMORY)
    if append_ub_check and src_smt_fun.invokes_ub is not None:
        solver.append(Not(src_smt_fun.invokes_ub))
    for constraint in src_smt_fun.state.ptr_constraints:
        solver.append(constraint)
    return solver


def show_solver_result(solver, transform_name, name, location, verbose):
    if verbose > 0:
        print(f"Checking {name}")
    if verbose > 1:
        print(solver.to_smt2())
    res = solver.check()
    if res == sat:
        if transform_name:
            msg = f"Transformation {transform_name}"
        else:
            msg = "Transformation"
        msg = msg + f" is not correct ({name})."
        gcc.inform(location, msg)
        return False, solver.model()
    if res != unsat:
        if transform_name:
            msg = f"Analysis of {transform_name} timed out ({name})"
        else:
            msg = f"Analysis timed out ({name})"
        gcc.inform(location, msg)
    return res != unsat, None


def check(
    src_smt_fun,
    tgt_smt_fun,
    location,
    report_success,
    verbose=0,
    transform_name="",
):
    if verbose > 0 and transform_name:
        print(f"{transform_name}:")

    src_retval = src_smt_fun.retval
    tgt_retval = tgt_smt_fun.retval

    # We check the return value before we check if tgt has more UB than src.
    # This is conceptually wrong, but it does not matter for correctness,
    # and it is somewhat more useful as we then can differentiate between
    # cases where we get a different value (either for incorrect calculations,
    # or because of new UB) and cases where we have new UB that does not
    # affect the result (i.e. much of the integer wrapping).

    # Check if return value is OK.
    success = True
    if src_retval is not None:
        solver = init_solver(src_smt_fun)
        solver.append(src_retval != tgt_retval)
        timeout, model = show_solver_result(
            solver, transform_name, "retval", location, verbose
        )
        success = success and not timeout
        if model is not None:
            msg = f"{model}\n"
            msg = msg + f"src retval: {model.eval(src_retval)}\n"
            msg = msg + f"tgt retval: {model.eval(tgt_retval)}"
            gcc.inform(location, msg)
            return

    # Check if tgt has more UB than src.
    if tgt_smt_fun.invokes_ub is not None:
        solver = init_solver(src_smt_fun)
        solver.append(tgt_smt_fun.invokes_ub)
        # We often have identical IR for src and tgt. Z3 is reasonably
        # good at seeing this and quickly return UNSAT. But it fails doing
        # this for some functions, and we in worst case end up with a
        # timeout for each pass (i.e., taking several hours to compile).
        # This issue goes away when we add a redundant src.ub != tgt.ub
        # check (but we keep the check for tgt.ub too -- it makes the
        # solver faster for other cases).
        solver.append(tgt_smt_fun.invokes_ub != src_smt_fun.invokes_ub)
        timeout, model = show_solver_result(
            solver, transform_name, "UB", location, verbose
        )
        success = success and not timeout
        if model is not None:
            msg = f"{model}\n"
            gcc.inform(location, msg)
            return

    # Check global memory.
    if src_smt_fun.state.global_memory:
        solver = init_solver(src_smt_fun)
        ptr = Const(".ptr", BitVecSort(64))
        mem_id = Extract(63, PTR_OFFSET_BITS, ptr)
        offset = Extract(PTR_OFFSET_BITS - 1, 0, ptr)
        valid_ptr = BoolVal(False)
        for mem_obj in src_smt_fun.state.global_memory:
            valid_id = mem_id == mem_obj.mem_id
            valid_offset = ULT(offset, Select(src_smt_fun.state.mem_sizes, mem_id))
            valid_ptr = Or(valid_ptr, And(valid_id, valid_offset))
        src_mem = src_smt_fun.memory
        tgt_mem = tgt_smt_fun.memory
        solver.append(valid_ptr)
        solver.append(Select(src_mem, ptr) != Select(tgt_mem, ptr))
        timeout, model = show_solver_result(
            solver, transform_name, "memory", location, verbose
        )
        success = success and not timeout
        if model is not None:
            msg = f"{model}\n"
            msg = msg + f"src *.ptr: {model.eval(Select(src_mem, ptr))}\n"
            msg = msg + f"tgt *.ptr: {model.eval(Select(tgt_mem, ptr))}"
            gcc.inform(location, msg)
            return

    if success and report_success:
        gcc.inform(location, "Transformation seems to be correct.")


def find_ub(smt_fun, location, verbose=0):
    success = True
    if smt_fun.invokes_ub is not None:
        solver = init_solver(smt_fun, False)
        solver.append(smt_fun.invokes_ub)
        if verbose > 1:
            print(solver.to_smt2())
        res = solver.check()
        if res == sat:
            msg = f"Invokes UB: {solver.model()}"
        elif res != unsat:
            msg = "Analysis timed out."
        else:
            msg = "Did not find any UB."
        gcc.inform(location, msg)
