const struct S {
  int a, b, c;
} s = {1,2,3};

int src(void)
{
  const struct S *p = &s;
  return p->b;
}

int tgt(void)
{
  return 2;
}
