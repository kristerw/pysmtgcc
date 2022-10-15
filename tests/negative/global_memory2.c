int a[3];

int src(int *p)
{
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  *p = 5;
  return a[1];
}

int tgt(int *p)
{
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  *p = 5;
  return 2;
}
