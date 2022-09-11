struct {
  int a[3];
  int b;
} s;

void src(int i)
{
  s.a[i] = 0;
  s.b = 1;
}

void tgt(int i)
{
  s.b = 1;
  s.a[i] = 0;
}
