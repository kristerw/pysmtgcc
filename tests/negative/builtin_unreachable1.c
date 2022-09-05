unsigned src(unsigned a)
{
  if (a > 24)
    __builtin_unreachable();
  return (256u << a) > 0;
}

unsigned tgt(unsigned a)
{
  return 1;
}
