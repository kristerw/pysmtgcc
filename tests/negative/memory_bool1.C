bool src(bool a)
{
  volatile bool t;
  t = a;
  return t;
}

bool tgt(bool a)
{
  return !a;
}
