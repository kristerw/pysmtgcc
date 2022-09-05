double src(void)
{
  union {
    long long i;
    double f;
  } u;
  u.i = 0x3ff0000000000000ll;
  return u.f;
}

double tgt(void)
{
  return 1.0;
}
