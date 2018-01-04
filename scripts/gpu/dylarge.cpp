//

// tensor.h
std::vector<int> count_larger(const Tensor& x, const Tensor& y, int dim);

// tensor.cc
template <class MyDevice> std::vector<int>
  count_larger_dev(const MyDevice& dev, const Tensor& x, const Tensor& y, int dim){
    throw std::runtime_error("max_and_argmax not implemented for this device.");
  }

#ifdef __CUDACC__

  /* impl_msp */

#include <assert.h>
  template <> std::vector<int>
    count_larger_dev<Device_GPU>(const Device_GPU& dev, const Tensor& x, const Tensor& y, int dim){
      if(dim > 0)
        DYNET_RUNTIME_ERR("Currently do not support dim > 1 in count_larger");
      DYNET_ARG_CHECK(x.mem_pool != DeviceMempool::NONE, "Input Tensor to max_and_argmax must be associated with a memory pool.");
      // todo(warn): need to consider broadcast ...
      int cur_size = x.d[0];
      int outer_size = x.d.size() / cur_size;
      //int inner_size = 1;
      vector<int> counts(outer_size);
      //
      int* gpu_counts;
      CUDA_CHECK(cudaSetDevice(((Device_GPU*)x.device)->cuda_device_id));
      CUDA_CHECK(cudaMalloc(&gpu_counts, sizeof(int)*outer_size));
      auto err0 = impl_msp::count_larger(x.v, y.v, gpu_counts, cur_size, outer_size, 1);
      CUDA_CHECK(err0);
      CUDA_CHECK(cudaMemcpy(counts.data(), gpu_counts, sizeof(int)*outer_size, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(gpu_counts));
      return counts;
    }

#else

#ifdef HAVE_CUDA
    // extern declaration
    extern template std::vector<int>
      count_larger_dev<Device_GPU>(const Device_GPU& dev, const Tensor& x, const Tensor& y, int dim);

    std::vector<int> count_larger(const Tensor& x, const Tensor& y, int dim) {
      if(x.device->type == DeviceType::CPU) { throw std::runtime_error("Bad device type"); }
      else if(x.device->type == DeviceType::GPU) { return count_larger_dev(*(const Device_GPU*)x.device, x, y, dim); }
      else { throw std::runtime_error("Bad device type"); }
    }
#else
  std::vector<int> count_larger(const Tensor& x, const Tensor& y, int dim) {
    throw std::runtime_error("Bad device type");
  }
#endif

#endif

// _dynet.pxd
from libcpp.pair cimport pair
cdef extern from "dynet/tensor.h" namespace "dynet":
    vector[int] c_count_larger "dynet::count_larger" (const CTensor& x, const CTensor& y, int dim)

// _dynet.pyx
cdef class Tensor:
    cpdef count_larger(self, Tensor y, int dim = 0) :
        return c_count_larger(self.t, y.t, dim)
