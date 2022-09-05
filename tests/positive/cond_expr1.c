int
src (int i)
{
  /* This is folded to a CondExpr in the frontend. */
  return 1 / i;
}

int
tgt (int i)
{
  if (i == 1)
    return 1;
  if (i == -1)
    return -1;
  return 0;
}
