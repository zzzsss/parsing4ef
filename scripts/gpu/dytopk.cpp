//

// tensor.h
std::pair<std::vector<int>, std::vector<real>> max_and_argmax(const Tensor& v, int dim, int k);

// tensor.cc
template <class MyDevice> std::pair<std::vector<int>, std::vector<real>>
max_and_argmax_dev(const MyDevice& dev, const Tensor& v, int dim, int k){ 
  throw std::runtime_error("max_and_argmax not implemented for this device.");
}

#ifdef __CUDACC__

/* tf_impl */

#include <assert.h>
template <> std::pair<std::vector<int>, std::vector<real>> 
max_and_argmax_dev<Device_GPU>(const Device_GPU& dev, const Tensor& v, int dim, int k){
  if(dim > 0)
    DYNET_RUNTIME_ERR("Currently do not support dim > 1 in max_and_argmax");
  DYNET_ARG_CHECK(v.mem_pool != DeviceMempool::NONE, "Input Tensor to max_and_argmax must be associated with a memory pool.");
  //
  int cur_size = v.d[0];
  int outer_size = v.d.size() / cur_size;
  int inner_size = 1;
  vector<int> idxs(outer_size*k);
  vector<real> vals(outer_size*k);
  //
  real* gpu_vals;
  int* gpu_idxs;
  CUDA_CHECK(cudaSetDevice(((Device_GPU*)v.device)->cuda_device_id));
  CUDA_CHECK(cudaMalloc(&gpu_vals, sizeof(real)*outer_size*k));
  CUDA_CHECK(cudaMalloc(&gpu_idxs, sizeof(int)*outer_size*k));
  auto err0 = impl_tf::topk<real, int, true>(v.v, gpu_vals, gpu_idxs, outer_size, inner_size, cur_size, k);
  CUDA_CHECK(err0);
  CUDA_CHECK(cudaMemcpy(vals.data(), gpu_vals, sizeof(real)*outer_size*k, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(idxs.data(), gpu_idxs, sizeof(int)*outer_size*k, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(gpu_vals));
  CUDA_CHECK(cudaFree(gpu_idxs));
  return std::make_pair(idxs, vals);
}

#else

template <> std::pair<std::vector<int>, std::vector<real>>
max_and_argmax_dev<Device_CPU>(const Device_CPU& dev, const Tensor& v, int dim, int k){
  throw std::runtime_error("max_and_argmax not implemented for CPU.");
}

#ifdef HAVE_CUDA
// extern declaration
extern template std::pair<std::vector<int>, std::vector<real>>
max_and_argmax_dev<Device_GPU>(const Device_GPU& dev, const Tensor& v, int dim, int k);

std::pair<std::vector<int>, std::vector<real>> max_and_argmax(const Tensor& v, int dim, int k) {
  if(v.device->type == DeviceType::CPU) { throw std::runtime_error("Bad device type"); }
  else if(v.device->type == DeviceType::GPU) { return max_and_argmax_dev(*(const Device_GPU*)v.device, v, dim, k); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
std::pair<std::vector<int>, std::vector<real>> max_and_argmax(const Tensor& v, int dim, int k){
  throw std::runtime_error("Bad device type");
}
#endif

#endif

// _dynet.pxd
from libcpp.pair cimport pair
cdef extern from "dynet/tensor.h" namespace "dynet":
    pair[vector[int], vector[float]] c_max_and_argmax "dynet::max_and_argmax" (CTensor& t, int dim, int k)

// _dynet.pyx
cdef class Tensor:
    cpdef max_and_argmax(self, int dim=0, int k=1):
        return c_max_and_argmax(self.t, dim, k)
