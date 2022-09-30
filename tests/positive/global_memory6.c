const char a[] = "abc";

char src(int i)
{
  if (i == 0)
    return 'a';
  if (i == 1)
    return 'b';
  if (i == 2)
    return 'c';
  if (i == 3)
    return 0;
  __builtin_unreachable();
}

char tgt(int i)
{
  return a[i];
}
