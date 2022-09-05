#include <stdint.h>

int8_t src(int8_t i)
{
  int8_t arr[3];
  arr[0] = 0;
  arr[1] = 1;
  arr[2] = 2;
  return arr[i];
}

int8_t tgt(int8_t i)
{
  return i;
}
