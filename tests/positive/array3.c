unsigned src(unsigned a, int b)
{
  unsigned arr[2];
  arr[0] = a;
  if (b > 0)
    arr[1] = 1;
  else
    arr[1] = 10;
  return arr[0] + arr[1];
}

unsigned tgt(unsigned a, int b)
{
  if (b > 0)
    return a + 1;
  else
    return a + 10;
}
