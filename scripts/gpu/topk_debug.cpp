#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
using namespace std;

namespace impl_tf{

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
  char shared_memory[48 << 10];
#define cudaError void

  // 
  enum class HeapType { kMinHeap, kMaxHeap };
  enum class PreferIndices { kLower, kHigher };   // prefer which index if equal

  template <typename T, typename IndexType>
  struct Entry {
    IndexType index;
    T value;
  };

  template <typename T, typename IndexType>
  struct LinearData {
    typedef impl_tf::Entry<T, IndexType> Entry;
    __device__ Entry& operator[](std::size_t index) const { return data[index]; }
    __device__ IndexType get_index(int i) const { return data[i].index; }
    __device__ T get_value(int i) const { return data[i].value; }
    Entry* const data;
  };

  template <typename T, typename IndexType>
  struct IndirectLinearData {
    typedef impl_tf::Entry<T, IndexType> Entry;
    __device__ Entry& operator[](std::size_t index) const { return data[index]; }
    __device__ IndexType get_index(int i) const { return backing_data[data[i].index].index; }
    __device__ T get_value(int i) const { return data[i].value; }
    Entry* const data;
    Entry* const backing_data;
  };

  template <typename T, typename IndexType>
  struct StridedData {
    typedef impl_tf::Entry<T, IndexType> Entry;
    __device__ Entry& operator[](std::size_t index) const { return data[index * blockDim.x + threadIdx.x]; }
    __device__ IndexType get_index(int i) const { return (*this)[i].index; }
    __device__ T get_value(int i) const { return (*this)[i].value; }
    Entry* const data;
  };


  // A heap of Entry<T> that can either work as a min-heap or as a max-heap.
  template <HeapType heapType, PreferIndices preferIndices, template <typename, typename> class Data, typename T, typename IndexType>
  struct IndexedHeap {
    typedef typename Data<T, IndexType>::Entry Entry;
    const Data<T, IndexType> data;

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

  template <HeapType heapType, PreferIndices preferIndices, template <typename, typename> class Data, typename T, typename IndexType>
  __device__ IndexedHeap<heapType, preferIndices, Data, T, IndexType> make_indexed_heap(
    typename Data<T, IndexType>::Entry* data) {
    return IndexedHeap<heapType, preferIndices, Data, T, IndexType>{Data<T, IndexType>{data}};
  }

  // heapTopK walks over [input, input+length) with `step_size` stride starting at
  // `start_index`.
  // It builds a top-`k` heap that is stored in `heap_entries` using `Accessor` to
  // access elements in `heap_entries`. If sorted=true, the elements will be
  // sorted at the end.
  template <typename T, typename IndexType, template <typename, typename> class Data = LinearData>
  __device__ void heapTopK(const T* __restrict__ input, IndexType length, IndexType k, Entry<T, IndexType>* __restrict__ heap_entries,
    bool sorted = false, IndexType start_index = 0, IndexType step_size = 1) {
    // this should be restricted previously
    //assert(k <= (length-start_index+step_size-1)/step_size);
    // the min value as the threshold
    auto heap = make_indexed_heap<HeapType::kMinHeap, PreferIndices::kHigher, Data, T, IndexType>(heap_entries);
    int heap_end_index = start_index + k * step_size;
    if(heap_end_index > length) {
      heap_end_index = length;
    }
    // Initialize the min-heap with the first k ones.
    for(IndexType index = start_index, slot = 0; index < heap_end_index; index += step_size, slot++) {
      heap.assign(slot, {index, input[index]});
    }
    heap.build(k);
    // Now iterate over the remaining items.
    // If an item is smaller than the min element, it is not amongst the top k.
    // Otherwise, replace the min element with it and push upwards.
    for(IndexType index = heap_end_index; index < length; index += step_size) {
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
  template <typename T, typename IndexType>
  __device__ void mergeShards(IndexType num_shards, IndexType k,
    Entry<T, IndexType>* __restrict__ entries, Entry<T, IndexType>* __restrict__ top_k_heap,
    T* top_k_values, IndexType* top_k_indices) {
    // If k < num_shards, we can use a min-heap with k elements to get the top k
    // of the sorted blocks.
    // If k > num_shards, we can initialize a min-heap with the top element from
    // each sorted block.
    const IndexType heap_size = k < num_shards ? k : num_shards;

    // Min-heap part.
    {
      auto min_heap = IndexedHeap<HeapType::kMinHeap, PreferIndices::kHigher,
        IndirectLinearData, T, IndexType>{IndirectLinearData<T, IndexType>{top_k_heap, entries}};
      // Initialize the heap as a min-heap.
      for(IndexType slot = 0; slot < heap_size; slot++) {
        min_heap.assign(slot, {slot, entries[slot].value});
      }
      min_heap.build(heap_size);

      // Now perform top k with the remaining shards (if num_shards > heap_size).
      for(IndexType shard = heap_size; shard < num_shards; shard++) {
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
        IndirectLinearData, T, IndexType>{
        IndirectLinearData<T, IndexType>{top_k_heap, entries}};
      // Heapify into a max heap.
      max_heap.build(heap_size);

      // Now extract the minimum k-1 times.
      // k is treated specially.
      const IndexType last_k = k - 1;
      for(IndexType rank = 0; rank < last_k; rank++) {
        const Entry<T, IndexType>& max_element = max_heap.root();
        top_k_values[rank] = max_element.value;
        IndexType shard_index = max_element.index;
        top_k_indices[rank] = entries[shard_index].index;
        IndexType next_shard_index = shard_index + num_shards;
        // For rank < k-1, each top k heap still contains at least 1 element,
        // so we can draw a replacement.
        max_heap.replace_root({next_shard_index, entries[next_shard_index].value},
          heap_size);
      }

      // rank == last_k.
      const Entry<T, IndexType>& max_element = max_heap.root();
      top_k_values[last_k] = max_element.value;
      IndexType shard_index = max_element.index;
      top_k_indices[last_k] = entries[shard_index].index;
    }
  }

  extern __shared__ char shared_memory[];

  template <typename T, typename IndexType>
  __global__ void TopKKernel(const T* input, IndexType length, IndexType k,
    T* output, IndexType* indices) {
    const IndexType batch_index = blockIdx.x;
    const T* batch_input = input + batch_index * length;

    const IndexType thread_index = threadIdx.x;
    const IndexType thread_count = blockDim.x;

    Entry<T, IndexType>* shared_entries = (Entry<T, IndexType>*)shared_memory;

    // heap-select with strided elements
    heapTopK<T, IndexType, StridedData>(batch_input, length, k, shared_entries, true,
      thread_index, thread_count);

    __syncthreads();
    if(thread_index == 0) {
      const IndexType offset = batch_index * k;
      auto batch_output = output + offset;
      auto batch_indices = indices + offset;
      Entry<T, IndexType>* top_k_heap = shared_entries + thread_count * k;

      // TODO(blackhc): Erich says: Performance can likely be improved
      // significantly by having the merge be done by multiple threads rather than
      // just one.  ModernGPU has some nice primitives that could help with this.
      mergeShards(thread_count, k, shared_entries, top_k_heap, batch_output,
        batch_indices);

      // debug
      /*
      for(int i = 0; i < thread_count+1; i++){
      printf("Print for heap %d\n", i);
      auto* head = shared_entries;
      for(int j = 0; j < k; j++)
      printf("Heap-%d-%d-%.3f\n", j, (head + i*k + j)->index, (head + i*k + j)->value);
      }
      */
    }
  }

  template<typename T, typename IndexType>
  IndexType get_num_shards(IndexType k, IndexType length){
    // This code assumes that k is small enough that the computation
    // fits inside shared memory (hard coded to 48KB).  In practice this
    // means k <= 3072 for T=float/int32 and k <= 2048 for T=double/int64.
    // The calculation is:
    //   shared_memory_size / (2 * (sizeof(int) + sizeof(T))) < k.
    // Use as many shards as possible.
    IndexType num_shards;
    constexpr auto shared_memory_size = 48 << 10;  // 48 KB
    const auto heap_size = k * sizeof(Entry<T, IndexType>);
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
    //std::cout << "Auto-setting num_shards=" << num_shards << std::endl;
    return num_shards;
  }

  template<typename T, typename IndexType, bool IsMax>
  cudaError topk(T* input, T* ouput_val, IndexType* ouput_idx,
    IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k,
    IndexType num_shards = 0){
    // stricted, but could be extended
    if(!IsMax)
      throw std::runtime_error("Not implemented with tf for n-argmin.");
    if(inner_size != 1)
      throw std::runtime_error("Not implemented with tf for strided version.");
    //
    if(num_shards <= 0)
      num_shards = get_num_shards<T, IndexType>(k, cur_size);
    auto shared_memory_size = (num_shards + 1) * k * sizeof(Entry<T, IndexType>);
    //TopKKernel<<<outer_size, num_shards, shared_memory_size>>>(input, cur_size, k, ouput_val, ouput_idx);
    //return cudaGetLastError();
    // executing
    for(int i = 0; i < outer_size; i++){
      for(int j = num_shards - 1; j >= 0; j--){
        blockIdx.x = i;
        threadIdx.x = j;
        blockDim.x = num_shards;
        TopKKernel(input, cur_size, k, ouput_val, ouput_idx);
      }
    }
  }
};

// testing the most common case: dimension-2 k-argmax
template<typename T, typename IndexType>
void test2(IndexType NUM_OUTPUT, IndexType NUM_BATCH, IndexType K, IndexType NUM_STEP){
  IndexType NUM_STRIDE = 1;
  // allocating
  size_t size_input = NUM_OUTPUT * NUM_BATCH * sizeof(T);
  size_t size_val = K * NUM_BATCH * sizeof(T);
  size_t size_idx = K * NUM_BATCH * sizeof(IndexType);
  vector<T> v_input(NUM_OUTPUT*NUM_BATCH);
  vector<T> v_val(K*NUM_BATCH);
  vector<IndexType> v_idx(K*NUM_BATCH);
  for(int i = 0; i < NUM_BATCH; i++){
    for(int j = 0; j < NUM_OUTPUT; j++)
      v_input[i*NUM_OUTPUT + j] = -1 * T(j);
    shuffle(v_input.begin() + i*NUM_OUTPUT, v_input.begin() + (i + 1)*NUM_OUTPUT, std::default_random_engine(0));
  }
  //
  cout << "--- Start a test: " << NUM_OUTPUT << ' ' << NUM_BATCH << ' ' << K << ' ' << NUM_STEP << endl;
  // alloc and copy
  // Allocate memory for each vector on GPU
  T *cv_input = v_input.data();
  T *cv_val = v_val.data();
  IndexType *cv_idx = v_idx.data();
  //
  try{
    cout << "== Test with impl_tf, auto-shard is " << impl_tf::get_num_shards<T, IndexType>(K, NUM_OUTPUT) << endl;
    for(IndexType num_shards : {0, 1, 2, 4, 8, 64}){
      double timek = 0;
      double timec = 0;
      // calculate
      for(int i = 0; i < NUM_STEP; i++){
        auto t1 = clock();
        impl_tf::topk<T, IndexType, true>(cv_input, cv_val, cv_idx, NUM_BATCH, NUM_STRIDE, NUM_OUTPUT, K, num_shards);
        auto t2 = clock();
        // copy back
        auto t3 = clock();
        timek += (t2 - t1);
        timec += (t3 - t2);
      }
      //check(NUM_OUTPUT, NUM_BATCH, K, answers, v_input, v_val, v_idx);
      cout << "Time for tf-" << num_shards << " is " << timek / CLOCKS_PER_SEC << "/" << timec / CLOCKS_PER_SEC << endl;
    }
  }
  catch(...){ cout << "error" << endl; }
  return;
}

int main()
{
  test2<float, long>(40, 1, 4, 1);
  return 0;
}

// the problem might be 
// 1. IndexType and fixed-int in the original tf-impl
// 2. if num_shards*k>length, there will be rubbish entries.
