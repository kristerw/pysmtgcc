struct S
{
  unsigned int a:4;
  unsigned int b:5;
  unsigned int c:11;
};

unsigned src(void)
{
  struct S s;
  int i;
  __builtin_memset(&s, 0, 4);
  s.b = 25;
  __builtin_memcpy(&i, &s, 4);
  return i;
}

unsigned tgt(void)
{
  return 0x190;
}
