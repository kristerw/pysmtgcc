float src(void)
{
  union {
    int i;
    float f;
  } u;
  u.i = 0x3f800000;
  return u.f;
}

float tgt(void)
{
  return 1.0;
}
