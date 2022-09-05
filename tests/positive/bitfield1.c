struct S
{
  unsigned int a:4;
  unsigned int b:5;
  unsigned int c:11;
};

unsigned src(void)
{
  struct S s;

  int i = 0x190;
  __builtin_memcpy(&s, &i, 4);
  return s.b;
}

unsigned tgt(void)
{
  return 25;
}
