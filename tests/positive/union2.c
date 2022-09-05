int src(void)
{
  union {
    int i;
    float f;
  } u;
  u.f = 1.0;
  return u.i;
}

int tgt(void)
{
  return 0x3f800000;
}
