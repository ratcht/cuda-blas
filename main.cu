#include "l1.cuh"

int main() {
  const int n = 10;
  const float a = 1;
  const float x[n] = {1,2,-3,4,5,6,7,8,-9,10};
  const int incx = 1;
  float y[n] = {2,1,2,1,2,1,2,1,2,1}; 
  const int incy = 1;

  //printArray(x, n);
  //printArray(y, n);


  sscal(n, 3, y, 1);
  printArray(y, n);


  return 0;
}