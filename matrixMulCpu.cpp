#include <iostream>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <algorithm>

#define MSIZE 1024

void multiply(int msize, float *a, float *b, float *c)
{ 
    int i,j,k;
    #pragma omp parallel for
    for(i=0; i<msize; ++i) {
        for( int k=0; k<msize; ++k) {
            for(int j=0; j<msize; ++j) {
                c[i * msize + j] += a[i * msize + k] * b[k * msize + j];
            }
        }
    } 
}

void init_array(int msize, float *array, float value)
{
    for(int i=0; i<msize; i++){
        for(int j=0; j<msize; j++){
            array[i * msize + j] = value;
        }
    }
}

int main(int argc, char **argv){
    int msize = MSIZE;
    float *a = (float *)malloc((msize * msize) * sizeof(float));
    float *b = (float *)malloc((msize * msize) * sizeof(float));
    float *c = (float *)malloc((msize * msize) * sizeof(float));
    
    init_array(msize, a, 1.0);
    init_array(msize, b, 0.01);
    init_array(msize, c, 0.0);
    printf("Start multiply\n");

    auto start = std::chrono::high_resolution_clock::now();
    
    multiply(msize, a, b, c);

    auto end = std::chrono::high_resolution_clock::now();

    auto int_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "funcSleep() elapsed time is " << int_s.count() << " ms )" << std::endl;

    return 0;
}