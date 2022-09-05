#include <stdint.h>

int32_t src(int32_t i)
{
  int32_t arr[3];
  arr[0] = 0;
  arr[1] = 1;
  arr[2] = 2;
  return arr[i];
}

int32_t tgt(int32_t i)
{
  return i;
}
