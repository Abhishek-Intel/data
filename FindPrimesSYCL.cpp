#include <CL/sycl.hpp>
#include "FindPrimesSYCL.h"

#include <array>
#include <iostream>
#include <numeric>
#include <vector>
#include <chrono>
#include "work.h"

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

using namespace std;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
class FindPrimeI;

float find_prime_s(work* w) {

  // need this for the lambda capture and sycl queue submit
  auto &VRI = w->VRI;
  auto N = w->VRI.size();
  auto niter = w->niter;
  auto nitems = w->nitems;
  
  cl::sycl::range<1> numOfItems{nitems};
  cl::sycl::buffer<cl::sycl::cl_short, 1> bufferR(VRI.data(), N);

  auto start = std::chrono::high_resolution_clock::now();

  cl::sycl::event event;

#ifdef __SYCL_DEVICE_ONLY__
  #define CONSTANT __attribute__((opencl_constant))
#else
  #define CONSTANT
#endif
  
  if (w->queueLock) { w->queueLock->lock(); }
  event = w->deviceQueue->submit([&](cl::sycl::handler& cgh) {
      auto accessorR = bufferR.template get_access<sycl_write>(cgh);
            
      auto k2 = [=](cl::sycl::item<1> item) {
        size_t maxstride = 1 + N/nitems;

        
        for (size_t istride=0; istride < maxstride; ++istride) {
          
          unsigned int number = istride*nitems + item.get_linear_id();
          
          static const CONSTANT char FMT[] =
            "n: %d  is: %d   ms: %d  ni: %d  id: %d\n";
          sycl::intel::experimental::printf(FMT, (int)number, (int)istride,
                                            (int)maxstride,
                                            (int)nitems,
                                            (int)item.get_linear_id());
          
          if (number < N) {
            for (size_t i=0; i<niter; ++i) {
              bool is_prime = !(number%2 == 0);
              const int upper_bound = cl::sycl::sqrt(1.0*number)+1;
              int k = 3;
              while (k < upper_bound && is_prime) {
                is_prime = !(number%k == 0);
                k += 2; // don't have to test even numbers
              }
              accessorR[number] = is_prime;
            }
          } else {
            break;
          }
        }
        
      };    
      cgh.parallel_for<class FindPrimeI>(numOfItems, k2);
    });
  if (w->queueLock) { w->queueLock->unlock(); }
  
  //  deviceQueue.wait();
  try {
    event.wait_and_throw();
  } catch ( sycl::exception const& e ) {
    std::cout << "Caught asynchronous SYCL exception:\n"
              << e.what() << std::endl;
  }

  auto stop = std::chrono::high_resolution_clock::now();

  auto submit_time = event.get_profiling_info<
      cl::sycl::info::event_profiling::command_submit>();
  auto start_time = event.get_profiling_info<
    cl::sycl::info::event_profiling::command_start>();
  auto end_time = event.get_profiling_info<
    cl::sycl::info::event_profiling::command_end>();

  w->start_time  = start_time;
  w->end_time    = end_time;
  w->submit_time = submit_time;
  w->start       = start;
  w->stop        = stop;

  // std::cout << "submit time: " << submission_time
  //           << std::endl;
  // std::cout << "execut time: " << execution_time
  //           <<    std::endl;  
  
  w->result = 0;
  for (auto &e : w->VRI) {
    if (e) {
      ++w->result;
    }
  }
  
  std::chrono::duration<double> diff {0};
  diff = (stop-start);

  return diff.count();

}
