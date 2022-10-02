const float f = 123.456;

float src(void)
{
  const float *p = &f;
  return *p;
}

float tgt(void)
{
  return 123.456;
}
