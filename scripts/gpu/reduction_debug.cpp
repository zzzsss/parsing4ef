#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <typeinfo>
#include <vector>

namespace impl_msp{
  // emulate cuda variable
  struct CudaVar{
    int x;
  };
  CudaVar blockDim;
  CudaVar threadIdx;
  CudaVar blockIdx;
#define __device__
#define __shared__
#define __restrict__
#define __global__
#define __forceinline__
#define __syncthreads()
#define cudaError void

  __shared__ int sdata[512];

  template<typename T, typename R, typename IndexType, typename F, unsigned int blockSize = 512>
  __global__ void ReduceKernel(const T* input_thread, const T* input_block, R* output, IndexType length, F func){
    constexpr auto SR = blockSize;
    IndexType bid = blockIdx.x;
    IndexType bdim = blockDim.x;
    IndexType tid = threadIdx.x;
    IndexType input_base = bid*length;
    T block_v = input_block[bid];

#define REDUCING_ONCE(stride) {if(tid < stride){sdata[tid] += sdata[tid+stride];}}

    sdata[tid] = 0;
    for(IndexType cur = tid; cur < length; cur += bdim){
      sdata[tid] = func(sdata[tid], input_thread[input_base + cur], block_v);
    }
    __syncthreads();

    if(blockSize >= 512) { if(tid < 256) { REDUCING_ONCE(256); } __syncthreads(); }
    if(blockSize >= 256) { if(tid < 128) { REDUCING_ONCE(128); } __syncthreads(); }
    if(blockSize >= 128) { if(tid < 64) { REDUCING_ONCE(64); } __syncthreads(); }
    if(tid < 32)  // inside one warp, no need to sync
    {
      if(blockSize >= 64) REDUCING_ONCE(32);
      if(blockSize >= 32) REDUCING_ONCE(16);
      if(blockSize >= 16) REDUCING_ONCE(8);
      if(blockSize >= 8) REDUCING_ONCE(4);
      if(blockSize >= 4) REDUCING_ONCE(2);
      if(blockSize >= 2) REDUCING_ONCE(1);
    }

#undef REDUCING_ONCE

    if(tid == 0)
      output[bid] = sdata[0];
  }

  template<typename T, typename R>
  __device__ R plusone_if_larger(R c, T x, T y){
    if(x > y)
      return c + 1;
    else
      return c;
  }

  template<typename T, typename R, typename IndexType>
  cudaError count_larger(const T* x, const T* y, R* output, IndexType length, IndexType outer_size, IndexType inner_size){
    constexpr unsigned int blockSize = 32;
    if(inner_size != 1)
      throw std::runtime_error("Not implemented with tf for strided version.");
    //ReduceKernel << <outer_size, 512 >> >(x, y, output, length, plusone_if_larger<T, R>);
    auto ff = plusone_if_larger<T, R>;
    for(int i = 0; i < outer_size; i++){
      for(int j = blockSize - 1; j >= 0; j--){
        blockIdx.x = i;
        threadIdx.x = j;
        blockDim.x = blockSize;
        ReduceKernel<T, R, int, decltype(ff), blockSize>(x, y, output, length, ff);
      }
    }
  }
};

template <typename T>
T get_rand(int bits){
  T x = 0.;
  for(int i = 0; i < bits; i++){
    x += T(rand()) / RAND_MAX;
    x *= 10;
  }
  if(T(rand()) / RAND_MAX > 0.5)
    x *= -1;
  return x;
}

template <typename T>
void test(int batch_size, int out_size, int steps)
{
  using namespace std;
  // allocating
  size_t size_x = batch_size * out_size * sizeof(T);
  size_t size_y = batch_size * sizeof(T);
  size_t size_out = batch_size * sizeof(int);
  auto v_x = new T[batch_size*out_size];
  auto v_y = new T[batch_size];
  auto v_out = new int[batch_size];
  for(int i = 0; i < batch_size; i++){
    v_y[i] = get_rand<T>(1);
    for(int j = 0; j < out_size; j++)
      v_x[i*out_size + j] = get_rand<T>(1);
  }
  //
  cout << "--- Start a test: T=" << typeid(T).name() << ' ' << batch_size << ' ' << out_size << ' ' << steps << endl;
  // test on cpu
  vector<int> v_gold(batch_size, 0);
  {
    cout << "== Test with cpu" << endl;
    auto t1 = clock();
    for(int s = 0; s < steps; s++){
      for(int i = 0; i < batch_size; i++){
        int count = 0;
        for(int j = 0; j < out_size; j++){
          if(v_x[i*out_size + j] > v_y[i])
            count++;
        }
        v_gold[i] = count;
      }
    }
    auto t2 = clock();
    double time_once = double(t2 - t1) / CLOCKS_PER_SEC;
    cout << "Time on cpu is " << time_once << endl;
  }
  // test on impl
  {
    cout << "== Test with gpu" << endl;
    auto t1 = clock();
    for(int s = 0; s < steps; s++){
      impl_msp::count_larger(v_x, v_y, v_out, out_size, batch_size, 1);
    }
    auto t2 = clock();
    // check
    for(int i = 0; i < batch_size; i++)
      if(v_gold[i] != v_out[i])
        cout << "!! Error on " << i << ":" << v_gold[i] << "||" << v_out[i] << endl;
    double time_once = double(t2 - t1) / CLOCKS_PER_SEC;
    cout << "Time on gpu is " << time_once << endl;
  }
}

void setup(int gpuid){
  srand(12345);
}

int main()
{
  setup(6);
  //
  test<float>(80, 50000, 10);
  return 0;
}

