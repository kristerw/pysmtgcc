# pysmtgcc
This is an experimental implementation of translation validation for GCC (similar to the LLVM [Alive2](https://github.com/AliveToolkit/alive2)). The blog post "[GCC Translation Validation](https://kristerw.github.io/2022/09/13/translation-validation/)" contains some background information, and the implementation is described in a series of blog posts:
1. [Writing a GCC plugin in Python](https://kristerw.github.io/2022/10/20/gcc-python-plugin/)
2. [Verifying GCC optimizations using an SMT solver](https://kristerw.github.io/2022/11/01/verifying-optimizations/)
3. TBD Memory
4. TBD Control flow

This implementation has been reasonably successful and has uncovered several bugs in GCC ([106513](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106513),
[106523](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106523),
[106744](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106744),
[106883](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106883),
[106884](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106884),
[106990](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106990),
[108625](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108625),
[109626](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109626)).

I'm currently implementing a new production-quality version in C++ (expected to be release during the first half of 2023), so this experimental implementation will not get any further improvements.

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
gcc -O3 -fno-strict-aliasing -c -fplugin=python -fplugin-arg-python-script=plugin1.py pr106523.c
```
gives us the output
```
pr106523.c: In function 'f7':
pr106523.c:1:15: note: Transformation ccp -> forwprop is not correct (retval).
    1 | unsigned char f7(unsigned char x, unsigned int y)
      |               ^~
pr106523.c:1:15: note: [y = 13, x = 198]
src retval: 24
tgt retval: 216
```
telling us that the `forwprop` pass miscompiled the function, so it now returns `216` instead of `24` when called as `f7(198, 13)`.


# Using plugin2.py
`plugin2.py` requires the translation unit to consist of two functions named `src` and `tgt`, and it verifies that `tgt` is a refinement of `src`.

For example, testing changing the order of signed addition
```c
int src(int a, int b, int c)
{
  return a + c + b;
}

int tgt(int a, int b, int c)
{
  return a + b + c;
}
```
by compiling as
```
gcc -O3 -fno-strict-aliasing -c -fplugin=python -fplugin-arg-python-script=plugin2.py example.c
```
gives us the output
```
example.c: In function 'tgt':
example.c:6:5: note: Transformation is not correct (UB).
    6 | int tgt(int a, int b, int c)
      |     ^~~
example.c:6:5: note: [c = 1793412222, a = 3429154139, b = 2508144171]
```
telling us that `tgt` invokes undefined behavior in cases where `src` does not,
and gives us an example of input where this happen (the values are, unfortunately, written as unsigned values. In this case, it means `[c = 1793412222, a = -865813157, b = -1786823125]`).

**Note**: `plugin2.py` works on the IR from the `ssa` pass, i.e., early enough that the compiler has not done many optimizations. But GCC does peephole optimizations earlier (even when compiling as `-O0`), so we need to prevent that from happening when testing such optimizations. The pre-GIMPLE optimizations are done one statement at a time, so we can disable the optimization by splitting the optimized pattern into two statements. For example, to check the following optimization
```
-(a - b)  ->  b - a
```
we can write the test as
```
int src(int a, int b)
{
  int t = a - b;
  return -t;
}

int tgt(int a, int b)
{
  return b - a;
}
```
Another way to verify such optimizations is to write the test in GIMPLE and pass the `-fgimple` flag to the compiler.

It is good practice to check with `-fdump-tree-ssa` that the IR used by the tool looks as expected. 

# Limitations
Some of the major limitations in the current version:
* Function calls are not implemented.
* Loops are not implemented.
* Support for memory operations is a bit shaky:
  * `malloc` etc. are not supported.
  * The tool often reports spurious memory-related errors unless `-fno-strict-aliasing` is passed to the compiler.
  * Pointer size/memory order is hardcoded as 64 bits/little-endian.
  * ...

Another annoying limitation is that GCC is doing folding (i.e., peephole optimizations) before running GIMPLE passes, so the tool will not find bugs in that code. Alive/Alive2 found several bugs in the LLVM equivalent `instcombine` pass, so it is likely that GCC also has bugs in its peephole optimizations.
