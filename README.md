# pysmtgcc
Some experiments with SMT solvers and GIMPLE IR

# Installing
### gcc-python-plugin
`pysmtgcc` needs a special version of the [`gcc-python-plugin`](https://github.com/davidmalcolm/gcc-python-plugin) that adds some APIs not supported in the original. The version is available in the [`pysmtgcc` branch](https://github.com/kristerw/gcc-python-plugin/tree/pysmtgcc) in my fork of `gcc-python-plugin`.

When building, you need to specify the compiler that will use the plugin. I'm typically using the development version of GCC installed in `/scratch/gcc-trunk/install`, so I fetch, build, and install the python plugin as
```
git checkout -b pysmtgcc https://github.com/kristerw/gcc-python-plugin
cd gcc-python-plugin
make CC=/scratch/gcc-trunk/install/bin/gcc
make CC=/scratch/gcc-trunk/install/bin/gcc install
```
**Note**: There is a problem in Python 3.8 and most versions of 3.9 that makes the plugin fail with an error message that `gcc.WrapperMeta` is not iterable. If you get that error, you'll need to update to Python 3.9.13 or 3.10.

### Z3
You also need the [Z3](https://github.com/Z3Prover/z3) SMT solver and its Python bindings. The easiest way to install all of this is
```
pip3 install z3-solver
```

### pysmtgcc
The source code for `pysmtgcc` is available in this repository. No compilation is necessary – `plugin1.py` or `plugin2.py` is passed to GCC using a command line argument (see below).

# Using plugin1.py
`plugin1.py` compares the IR before/after each GIMPLE pass and complains if the resulting IR is not a refinement of the input (that is, if the GIMPLE pass miscompiled the program).

For example, compiling the function `f7` from [PR 106523](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106523)
```c
unsigned char f7(unsigned char x, unsigned int y)
{
  unsigned int t = x;
  return (t << y) | (t >> ((-y) & 7));
}
```
with a compiler where PR 106523 is not fixed (for example, GCC 12.2) using `plugin1.py`
```
gcc -O3 -c -fplugin=python -fplugin-arg-python-script=plugin1.py pr106523.c
```
gives us the output
```
pr106523.c: In function 'f7':
pr106523.c:1:15: note: Transformation ccp -> forwprop is not correct (retval).
[y = 13, x = 198]
```
telling us that the `forwprop` pass miscompiled the function so that it now returns a different value when called as `f7(198, 13)`.


# Using plugin2.py
TBD
