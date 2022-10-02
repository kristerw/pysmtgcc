const long double f = 123.456;

long double src(void)
{
  const long double *p = &f;
  return *p;
}

long double tgt(void)
{
  return 123.456;
}
