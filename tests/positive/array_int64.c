#include <stdint.h>

int64_t src(int64_t i)
{
  int64_t arr[3];
  arr[0] = 0;
  arr[1] = 1;
  arr[2] = 2;
  return arr[i];
}

int64_t tgt(int64_t i)
{
  return i;
}
