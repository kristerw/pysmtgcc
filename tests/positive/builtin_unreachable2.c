unsigned src(unsigned a)
{
  __builtin_unreachable();
  return 2;
}

unsigned tgt(unsigned a)
{
  return 1;
}
