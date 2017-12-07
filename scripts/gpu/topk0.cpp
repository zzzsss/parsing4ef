#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <ctime>

/*
#define __device__
#define __shared__
#define __restrict__
#define __global__
*/

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

// using eigen
namespace impl_eigen{
  cudaError topk(int out_size, int batch_size, float* v_input, float* v_val, long* v_idx, int k, Eigen::GpuDevice& my_device){
    if(k != 1)
      throw std::runtime_error("Not implemented for k != 1.");
    (Eigen::TensorMap<Eigen::Tensor<long, 1>>(v_idx, batch_size)).device(my_device)
      = (Eigen::TensorMap<Eigen::Tensor<float, 2>>(v_input, out_size, batch_size)).argmax(0);
    return cudaGetLastError();
  }
};

namespace impl_tr{
  template <typename T>
  struct TopKTypeConfig {};

  template <>
  struct TopKTypeConfig<float> {
    typedef uint32_t RadixType;
    // Converts a float to an integer representation with the same
    // sorting; i.e., for floats f1, f2:
    // if f1 < f2 then convert(f1) < convert(f2)
    // We use this to enable radix selection of floating-point values.
    // This also gives a relative order for NaNs, but that's ok, as they
    // will all be adjacent
    static inline __device__ RadixType convert(float v) {
      RadixType x = __float_as_int(v);
      RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

      return (x ^ mask);
    }
    static inline __device__ float deconvert(RadixType v) {
      RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

      return __int_as_float(v ^ mask);
    }
  };

  // the interface
  template<typename T, typename IndexType>
  cudaError topk(T* input, ){
    return cudaGetLastError();
  }

};

// from tensorflow's implementation
namespace impl_tf {

  enum class HeapType { kMinHeap, kMaxHeap };
  enum class PreferIndices { kLower, kHigher };   // prefer which index if equal

  template <typename T>
  struct Entry {
    int index;
    T value;
  };

  template <typename T>
  struct LinearData {
    typedef impl_tf::Entry<T> Entry;
    __device__ Entry& operator[](std::size_t index) const { return data[index]; }
    __device__ int get_index(int i) const { return data[i].index; }
    __device__ T get_value(int i) const { return data[i].value; }
    Entry* const data;
  };

  template <typename T>
  struct IndirectLinearData {
    typedef impl_tf::Entry<T> Entry;
    __device__ Entry& operator[](std::size_t index) const { return data[index]; }
    __device__ int get_index(int i) const { return backing_data[data[i].index].index; }
    __device__ T get_value(int i) const { return data[i].value; }
    Entry* const data;
    Entry* const backing_data;
  };

  template <typename T>
  struct StridedData {
    typedef impl_tf::Entry<T> Entry;
    __device__ Entry& operator[](std::size_t index) const { return data[index * blockDim.x + threadIdx.x]; }
    __device__ int get_index(int i) const { return (*this)[i].index; }
    __device__ T get_value(int i) const { return (*this)[i].value; }
    Entry* const data;
  };


  // A heap of Entry<T> that can either work as a min-heap or as a max-heap.
  template <HeapType heapType, PreferIndices preferIndices, template <typename> class Data, typename T>
  struct IndexedHeap {
    typedef typename Data<T>::Entry Entry;
    const Data<T> data;

    // indicating whether left should be prior to right
    __device__ bool is_above(int left, int right) {
      T left_value = data.get_value(left);
      T right_value = data.get_value(right);
      if(left_value == right_value) {
        if(preferIndices == PreferIndices::kLower) {
          return data.get_index(left) < data.get_index(right);
        }
        else {
          return data.get_index(left) > data.get_index(right);
        }
      }
      if(heapType == HeapType::kMinHeap) {
        return left_value < right_value;
      }
      else {
        return left_value > right_value;
      }
    }

    // assign one entry
    __device__ void assign(int i, const Entry& entry) { data[i] = entry; }

    // swap
    __device__ void swap(int a, int b) {
      auto tmp = data[b];
      data[b] = data[a];
      data[a] = tmp;
    }

    // upward from i
    __device__ void push_up(int i) {
      int child = i;
      int parent;
      for(; child > 0; child = parent) {
        parent = (child - 1) / 2;
        if(!is_above(child, parent)) {
          // Heap property satisfied.
          break;
        }
        swap(child, parent);
      }
    }

    __device__ void push_root_down(int k) { push_down(0, k); }

    // MAX-HEAPIFY in Cormen, k is the range
    __device__ void push_down(int node, int k) {
      while(true) {
        const int left = 2 * node + 1;
        const int right = left + 1;
        int smallest = node;
        if(left < k && is_above(left, smallest)) {
          smallest = left;
        }
        if(right < k && is_above(right, smallest)) {
          smallest = right;
        }
        if(smallest == node) {
          break;
        }
        swap(smallest, node);
        node = smallest;
      }
    }

    // BUILD-MAX-HEAPIFY in Cormen
    __device__ void build(int k) {
      for(int node = (k - 1) / 2; node >= 0; node--) {
        push_down(node, k);
      }
    }

    // HEAP-EXTRACT-MAX in Cormen
    __device__ void remove_root(int k) {
      data[0] = data[k - 1];
      push_root_down(k - 1);
    }

    // in-place HEAPSORT in Cormen (turn minHeap to max-sorting)
    // This method destroys the heap property.
    __device__ void sort(int k) {
      for(int slot = k - 1; slot > 0; slot--) {
        // This is like remove_root but we insert the element at the end.
        swap(slot, 0);
        // Heap is now an element smaller.
        push_root_down(/*k=*/slot);
      }
    }

    __device__ void replace_root(const Entry& entry, int k) {
      data[0] = entry;
      push_root_down(k);
    }

    __device__ const Entry& root() { return data[0]; }
  };

  template <HeapType heapType, PreferIndices preferIndices,
    template <typename> class Data, typename T>
  __device__ IndexedHeap<heapType, preferIndices, Data, T> make_indexed_heap(
    typename Data<T>::Entry* data) {
    return IndexedHeap<heapType, preferIndices, Data, T>{Data<T>{data}};
  }

  // heapTopK walks over [input, input+length) with `step_size` stride starting at
  // `start_index`.
  // It builds a top-`k` heap that is stored in `heap_entries` using `Accessor` to
  // access elements in `heap_entries`. If sorted=true, the elements will be
  // sorted at the end.
  template <typename T, template <typename> class Data = LinearData>
  __device__ void heapTopK(const T* __restrict__ input, int length, int k, Entry<T>* __restrict__ heap_entries,
    bool sorted = false, int start_index = 0, int step_size = 1) {
    // this should be restricted previously
    //assert(k <= (length-start_index+step_size-1)/step_size);
    // the min value as the threshold
    auto heap = make_indexed_heap<HeapType::kMinHeap, PreferIndices::kHigher, Data, T>(heap_entries);
    int heap_end_index = start_index + k * step_size;
    if(heap_end_index > length) {
      heap_end_index = length;
    }
    // Initialize the min-heap with the first k ones.
    for(int index = start_index, slot = 0; index < heap_end_index; index += step_size, slot++) {
      heap.assign(slot, {index, input[index]});
    }
    heap.build(k);
    // Now iterate over the remaining items.
    // If an item is smaller than the min element, it is not amongst the top k.
    // Otherwise, replace the min element with it and push upwards.
    for(int index = heap_end_index; index < length; index += step_size) {
      // We prefer elements with lower indices. This is given here.
      // Later elements automatically have higher indices, so can be discarded.
      if(input[index] > heap.root().value) {
        // This element should replace the min.
        heap.replace_root({index, input[index]}, k);
      }
    }
    // Sort if wanted.
    if(sorted) {
      heap.sort(k);
    }
  }

  // mergeShards performs a top-k merge on `num_shards` many sorted streams that
  // are sorted and stored in `entries` in a strided way:
  // |s_1 1st|s_2 1st|...s_{num_shards} 1st|s_1 2nd|s_2 2nd|...
  // The overall top k elements are written to `top_k_values` and their indices
  // to top_k_indices.
  // `top_k_heap` is used as temporary storage for the merge heap.
  template <typename T>
  __device__ void mergeShards(int num_shards, int k,
    Entry<T>* __restrict__ entries,
    Entry<T>* __restrict__ top_k_heap, T* top_k_values,
    int* top_k_indices) {
    // If k < num_shards, we can use a min-heap with k elements to get the top k
    // of the sorted blocks.
    // If k > num_shards, we can initialize a min-heap with the top element from
    // each sorted block.
    const int heap_size = k < num_shards ? k : num_shards;

    // Min-heap part.
    {
      auto min_heap = IndexedHeap<HeapType::kMinHeap, PreferIndices::kHigher,
        IndirectLinearData, T>{
        IndirectLinearData<T>{top_k_heap, entries}};
      // Initialize the heap as a min-heap.
      for(int slot = 0; slot < heap_size; slot++) {
        min_heap.assign(slot, {slot, entries[slot].value});
      }
      min_heap.build(heap_size);

      // Now perform top k with the remaining shards (if num_shards > heap_size).
      for(int shard = heap_size; shard < num_shards; shard++) {
        const auto entry = entries[shard];
        const auto root = min_heap.root();
        if(entry.value < root.value) {
          continue;
        }
        if(entry.value == root.value &&
          entry.index > entries[root.index].index) {
          continue;
        }
        // This element should replace the min.
        min_heap.replace_root({shard, entry.value}, heap_size);
      }
    }

    // Max-part.
    {
      // Turn the min-heap into a max-heap in-place.
      auto max_heap = IndexedHeap<HeapType::kMaxHeap, PreferIndices::kLower,
        IndirectLinearData, T>{
        IndirectLinearData<T>{top_k_heap, entries}};
      // Heapify into a max heap.
      max_heap.build(heap_size);

      // Now extract the minimum k-1 times.
      // k is treated specially.
      const int last_k = k - 1;
      for(int rank = 0; rank < last_k; rank++) {
        const Entry<T>& max_element = max_heap.root();
        top_k_values[rank] = max_element.value;
        int shard_index = max_element.index;
        top_k_indices[rank] = entries[shard_index].index;
        int next_shard_index = shard_index + num_shards;
        // For rank < k-1, each top k heap still contains at least 1 element,
        // so we can draw a replacement.
        max_heap.replace_root({next_shard_index, entries[next_shard_index].value},
          heap_size);
      }

      // rank == last_k.
      const Entry<T>& max_element = max_heap.root();
      top_k_values[last_k] = max_element.value;
      int shard_index = max_element.index;
      top_k_indices[last_k] = entries[shard_index].index;
    }
  }

  extern __shared__ char shared_memory[];

  template <typename T>
  __global__ void TopKKernel(const T* input, int length, int k, bool sorted,
    T* output, int* indices) {
    const int batch_index = blockIdx.x;
    const T* batch_input = input + batch_index * length;

    const int thread_index = threadIdx.x;
    const int thread_count = blockDim.x;

    Entry<T>* shared_entries = (Entry<T>*)shared_memory;

    // heap-select with strided elements
    heapTopK<T, StridedData>(batch_input, length, k, shared_entries, true,
      thread_index, thread_count);

    __syncthreads();
    if(thread_index == 0) {
      const int offset = batch_index * k;
      auto batch_output = output + offset;
      auto batch_indices = indices + offset;
      Entry<T>* top_k_heap = shared_entries + thread_count * k;

      // TODO(blackhc): Erich says: Performance can likely be improved
      // significantly by having the merge be done by multiple threads rather than
      // just one.  ModernGPU has some nice primitives that could help with this.
      mergeShards(thread_count, k, shared_entries, top_k_heap, batch_output,
        batch_indices);
    }
  }

  template <typename T>
  cudaError LaunchTopKKernel(int num_shards, const T* input, int batch_size, int length, int k, 
    bool sorted, T* output, int* indices) {
    // This code assumes that k is small enough that the computation
    // fits inside shared memory (hard coded to 48KB).  In practice this
    // means k <= 3072 for T=float/int32 and k <= 2048 for T=double/int64.
    // The calculation is:
    //   shared_memory_size / (2 * (sizeof(int) + sizeof(T))) < k.
    // Use as many shards as possible.
    if(num_shards <= 0) {
      constexpr auto shared_memory_size = 48 << 10;  // 48 KB
      const auto heap_size = k * sizeof(Entry<T>);
      // shared_memory_size = (num_shards + 1) * heap_size <=>
      num_shards = shared_memory_size / heap_size - 1;
      if(num_shards <= 0) {
        num_shards = 1;
      }
      auto shard_size = length / num_shards;
      auto min_shard_size = 2 * k;
      if(shard_size < min_shard_size) {
        num_shards = length / min_shard_size;
      }
      if(num_shards <= 0) {
        num_shards = 1;
      }
      else if(num_shards > 1024) {
        num_shards = 1024;
      }
    }
    //std::cout << "Calling with " << batch_size << "/" << num_shards << std::endl;
    // We are limited by the amount of shared memory we have per block.
    auto shared_memory_size = (num_shards + 1) * k * sizeof(Entry<T>);
    TopKKernel<<<batch_size, num_shards, shared_memory_size>>>(
      input, length, k, sorted, output, indices);
    return cudaGetLastError();
  }
};

#define CUDA_CHECK(stmt) do {                              \
    cudaError_t err = stmt;                                \
    if (err != cudaSuccess) {                              \
      std::cerr << "CUDA failure in " << #stmt << std::endl\
                << cudaGetErrorString(err) << std::endl;   \
      throw std::runtime_error(#stmt);                  \
    }                                                      \
  } while(0)

void test(){
  // test with them
  using std::cout;
  using std::endl;
  using std::clock;
  //
  int DEVICE_ID = 7;
  CUDA_CHECK(cudaSetDevice(DEVICE_ID));
  srand(12345);
  // allocating
  int NUM_STEP = 50;
  int NUM_OUTPUT = 50000;
  int NUM_BATCH = 80;
  int k = 1;

  size_t size_input = NUM_OUTPUT * NUM_BATCH * sizeof(float);
  size_t size_val = k * NUM_BATCH * sizeof(float);
  size_t size_idx = k * NUM_BATCH * sizeof(int);
  size_t size_idx_long = k * NUM_BATCH * sizeof(long);

  auto v_input = new float[NUM_OUTPUT*NUM_BATCH];
  auto v_val = new float[k*NUM_BATCH];
  auto v_idx = new int[k*NUM_BATCH];
  auto v_idx_long = new long[k*NUM_BATCH];
  cout << "Random selecting from " << RAND_MAX << endl;
  for(int i = 0; i < NUM_OUTPUT*NUM_BATCH; i++)
    v_input[i] = (rand()+0.f) / RAND_MAX;

  // alloc and copy
  // Allocate memory for each vector on GPU
  float *cv_input;
  float *cv_val;
  int *cv_idx;
  long *cv_idx_long;
  cudaMalloc(&cv_input, size_input);
  cudaMalloc(&cv_val, size_val);
  cudaMalloc(&cv_idx, size_idx);
  cudaMalloc(&cv_idx_long, size_idx_long);
  cudaMemcpy(cv_input, v_input, size_input, cudaMemcpyHostToDevice);

  cout << "Test with impl_tf" << endl;
  for(int num_shreds = 0; num_shreds < 1000; num_shreds+=100){
    // calculate
    auto t1 = clock();
    for(int i = 0; i < NUM_STEP; i++){
      CUDA_CHECK(impl_tf::LaunchTopKKernel(num_shreds, cv_input, NUM_BATCH, NUM_OUTPUT, k, true, cv_val, cv_idx));
      // copy back
      cudaMemcpy(v_val, cv_val, size_val, cudaMemcpyDeviceToHost);
      cudaMemcpy(v_idx, cv_idx, size_idx, cudaMemcpyDeviceToHost);
    }
    auto t2 = clock();
    cout << "Time for tf " << num_shreds << " is " << double(t2 - t1) / CLOCKS_PER_SEC << endl;
  }
  cout << "Test with impl_eigen" << endl;
  auto estream = new Eigen::CudaStreamDevice(DEVICE_ID);
  auto edevice = new Eigen::GpuDevice(estream);
  // calculate
  auto t1 = clock();
  for(int i = 0; i < NUM_STEP; i++){
    CUDA_CHECK(impl_eigen::topk(NUM_OUTPUT, NUM_BATCH, cv_input, cv_val, cv_idx_long, k, *edevice));
    // copy back
    cudaMemcpy(v_val, cv_val, size_val, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_idx, cv_idx, size_idx, cudaMemcpyDeviceToHost);
  }
  auto t2 = clock();
  cout << "Time for eigen is " << double(t2 - t1) / CLOCKS_PER_SEC << endl;
  // free
  cudaFree(cv_input);
  cudaFree(cv_val);
  cudaFree(cv_idx);
  return;
}

int main()
{
  test();
  return 0;
}

// compile: nvcc -std=c++11 -DEIGEN_USE_GPU -I../libs/eigen-eigen-5a0156e40feb-334/ ./t.cu
