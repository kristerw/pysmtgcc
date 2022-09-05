int src(int i, int a)
{
  unsigned arr[3];
  arr[i] = a;
  return arr[i];
}

int tgt(int i, int a)
{
  return a;
}
