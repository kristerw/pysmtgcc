float src(float a)
{
  volatile float t;
  t = a;
  return t;
}

float tgt(float a)
{
  return a;
}
