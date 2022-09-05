struct S
{
  int a:4;
  int b:5;
  int c:11;
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
  return 0xfffffff9;
}
