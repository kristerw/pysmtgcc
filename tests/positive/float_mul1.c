// Floating point is slow, so just do a trivial test to verify that the
// IR can be parsed.

float src(float a, float b)
{
  return a * b;
}

float tgt(float a, float b)
{
  return a * b;
}
