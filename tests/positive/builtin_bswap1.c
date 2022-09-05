unsigned src(unsigned a)
{
  return __builtin_bswap32(a);
}

unsigned tgt(unsigned a)
{
  return ((a & 0x000000ff) << 24 |
	  (a & 0x0000ff00) <<  8 |
	  (a & 0x00ff0000) >>  8 |
	  (a & 0xff000000) >> 24);
}
