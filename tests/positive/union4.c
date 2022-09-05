long long src(void)
{
  union {
    long long i;
    double f;
  } u;
  u.f = 1.0;
  return u.i;
}

long long tgt(void)
{
  return 0x3ff0000000000000;
}
