int src(int i)
{
  float f = 1.0;
  char *p = (char *)&f;
  return p[i];
}

int tgt(int i)
{
  if (i == 0)
    return 0;
  if (i == 1)
    return 0;
  if (i == 2)
    return -128;
  return 63;
}
