int src(int a)
{
  volatile int t;
  t = a;
  return t;
}

int tgt(int a)
{
  return a;
}
