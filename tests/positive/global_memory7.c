int src(int i)
{
  static int a[] = {0, 2, 4, 6, 8};
  return a[i];
}

int tgt(int i)
{
  return 2 * i;
}
