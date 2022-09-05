struct S
{
  unsigned int a:24;
};

unsigned src(void)
{
  struct S s;
  int i;
  __builtin_memset(&s, 0, 4);
  s.a = 25;
  __builtin_memcpy(&i, &s, 4);
  return i;
}

unsigned tgt(void)
{
  return 25;
}
