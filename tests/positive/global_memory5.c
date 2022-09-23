const int a[] = {0, 2, 4, 6, 8};

int src(int i, int *p)
{
  *p = 1;
  return a[i];
}

int tgt(int i, int *p)
{
  *p = 1;
  return 2 * i;
}
