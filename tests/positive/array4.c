int src(int i, int a)
{
  unsigned arr[3];
  arr[i] = 1;
  if (a)
    arr[i] = 2;
  return arr[i];
}

int tgt(int i, int a)
{
  if (a)
    return 2;
  return 1;
}
