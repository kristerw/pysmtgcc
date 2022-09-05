unsigned src(unsigned a)
{
  struct {
    unsigned a;
    unsigned b;
  } s;
  s.a = a;
  s.b = 1;
  return s.a + s.b;
}

unsigned tgt(unsigned a)
{
  return a + 1;
}
