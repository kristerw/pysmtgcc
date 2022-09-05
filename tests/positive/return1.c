unsigned src(unsigned a, unsigned b)
{
  if (a == 0)
    return b;
  if (a == 4)
    return b + 4;
  return a + b;
}

unsigned tgt(unsigned a, unsigned b)
{
  return a + b;
}
