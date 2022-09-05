int src(int i, int a)
{
  unsigned arr[10];
  unsigned *p = &arr[2];
  p[i] = a;
  return arr[i+2];
}

int tgt(int i, int a)
{
  return a;
}
