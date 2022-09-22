const int a = 2;

int src(void)
{
  const int *p = &a;
  return *p;
}

int tgt(void)
{
  return 2;
}
