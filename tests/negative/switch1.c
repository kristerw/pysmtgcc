int src(int a)
{
  switch(a)
    {
    case 1:
      return 5;
    case 2:
      return 23;
    case 5 ... 8:
	return 42;
    default:
      return 0;
    }
}

int tgt(int a)
{
  if (a == 1)
    return 5;
  if (a == 2)
    return 23;
  if (5 <= a && a <= 7)
    return 42;
  return 0;
}
