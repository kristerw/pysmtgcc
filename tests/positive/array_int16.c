#include <stdint.h>

int16_t src(int16_t i)
{
  int16_t arr[3];
  arr[0] = 0;
  arr[1] = 1;
  arr[2] = 2;
  return arr[i];
}

int16_t tgt(int16_t i)
{
  return i;
}
