unsigned src(unsigned a, unsigned b, unsigned c)
{
  return (a & ~c) | (b & c);
}

unsigned tgt(unsigned a, unsigned b, unsigned c)
{
  return a ^ ((a ^ b) & c);
}
