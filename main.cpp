#include <CL/sycl.hpp>
#include "FindPrimesSYCL.h"
#include "Crunch.h"
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <functional>
#include <vector>
#include <cstdlib>
#include <atomic>
#include <thread>
#include "MSG.h"
#include "work.h"


using namespace std;
namespace sycl = cl::sycl;

class gpu_tsk {
public:
  gpu_tsk(work& w) : m_work(w){
  }
  void operator() () const {

    MSG("GPU start: max prime: " << m_work.size << "  " << m_work.niter << " iters");
    find_prime_s(&m_work);

    m_work.run = (m_work.end_time - m_work.start_time)/ 1000000000.0f;
    m_work.wait = (m_work.start_time - m_work.submit_time)/ 1000000000.0f;
    m_work.elapsed = m_work.run + m_work.wait;
    MSG( "GPU done: " << m_work.elapsed << "  wait: "
         << m_work.wait << "  run: " << m_work.run );
        
  }
private:
  work &m_work;
};


static std::atomic<float> cpusum {0.};

class cpu_tsk {
public:
  cpu_tsk(int niter, Crunch& c) : m_n(niter), m_c(c) {}
  void operator() () const {

    MSG("CPU start: " << m_n << " iters");
    std::chrono::duration<double> diff {0};
    auto start = std::chrono::high_resolution_clock::now();
    m_c.crunch(m_n);
    auto stop = std::chrono::high_resolution_clock::now();
    diff = (stop-start);
    //    MSG("CPU end: " << m_n << "  t: " << t << "  r: " << float(m_n)/t);
    MSG("CPU end: " << diff.count() );
    cpusum = cpusum + (float)diff.count();
    
  }
private:
  int m_n;
  Crunch &m_c;
};

bool sortBySubmitTime(work& w1, work &w2) { return w1.submit_time < w2.submit_time; }


int main(int argc, char* argv[]) {

  int opt;
  bool do_gpu{ false };
  bool do_cpu{ false };
  bool help{false};
  bool lock{false};
  bool sharedQueue{false};
  bool showinfo{false};
  int nthreadsCPU = 1;
  int nthreadsGPU = 1;
  int arr_size = 20;
  int iter_gpu = 200;
  int iter_cpu = 100000;
  unsigned int gpu_dev = 999;
  unsigned int nitems = 0;

  while ((opt = getopt(argc, argv, "hs:t:T:cgC:G:D:iln:S")) != -1) {
    switch (opt) {
    case 's':
      arr_size = atoi(optarg);
      break;
    case 't':
      nthreadsCPU = atoi(optarg);
      break;
    case 'T':
      nthreadsGPU = atoi(optarg);
      break;
    case 'c':
      do_cpu = true;
      break;
    case 'g':
      do_gpu = true;
      break;
    case 'C':
      do_cpu = true;
      iter_cpu = atoi(optarg);
      break;
    case 'G':
      do_gpu = true;
      iter_gpu = atoi(optarg);
      break;
    case 'D':
      gpu_dev = atoi(optarg);
      break;
    case 'S':
      sharedQueue = true;
      break;
    case 'l':
      lock = true;
      break;
    case 'n':
      nitems = atoi(optarg);
      break;
    case 'i':
      showinfo = true;
      break;
    case 'h':
      help = true;
    default: /* '?' */
      fprintf(stderr, "Usage: %s [-hcgilS] [-t NTHREADS_CPU] [-t NTHREADS_GPU] [-s ARRAY_SIZE] [-C ITER_CPU] [-G ITER_GPU] [-n NITEMS] [-D CL_DEVICE]\n\n",
              argv[0]);
      if (help) {
        printf("         -c : do CPU run\n");
        printf("         -g : do GPU run\n\n");
        printf("         -l : lock the shared cl::sycl::queue with a mutex\n");
        printf("         -t N : spawn N threads for CPU work. Default is %d\n",
               nthreadsCPU);
        printf("         -T N : spawn N threads for GPU work. Default is %d\n",
               nthreadsGPU);
        printf("         -C N : do N iters of CPU work. Default is %d\n", iter_cpu);
        printf("         -G N : do N iters of GPU work. Default is %d\n\n", iter_gpu);
        printf("         -S : use single shared sycl::queue for GPU\n");
        printf("         -s N : use array of size 1<<N (max prime size) for GPU work. Default is %d (max prime: %d)\n",arr_size, 1<<arr_size);
        printf("         -n N : max SyCL items\n\n");
        printf("         -D N : use OpenCL device N\n");
        printf("         -i : show device info and exit\n");
          

      }

      exit(EXIT_FAILURE);
    }
  }
  
  // Size of arrays
  size_t N = 1<<arr_size;

  sycl::device sel_dev;
  std::mutex* queueLock {nullptr};

  if (lock) {
    queueLock = new std::mutex;
  }

  if (nitems == 0) {
    nitems = N;
  }
  if (nitems > N) {
    nitems = N;
  }
  
  std::cout << "NThreads: CPU " << nthreadsCPU << "   GPU " << nthreadsGPU << std::endl;
  
  std::cout << "do GPU: " << std::boolalpha << do_gpu << std::endl;
  if (do_gpu || showinfo) {
    cout << "   shared Queue: " << std::boolalpha << sharedQueue << endl;
    cout << "   array size: " << N << " (1<<" << arr_size << ")" << endl;
    cout << "   SyCL items: " << nitems << endl;
    cout << "   iter GPU:   " << iter_gpu << "\n";

    auto platform_list = sycl::platform::get_platforms();

    std::cout << "   available devices [" << platform_list.size() << "]\n";
    std::vector<sycl::device> dlist;
    
    // looping over platforms
    int i=0;
    for (const auto &platform : platform_list) {
      cout << "    [" << i << "] "
           << platform.get_info<sycl::info::platform::name>() << endl;
      // getting the list of devices from the platform
      auto device_list = platform.get_devices();
      // looping over devices
      for (const auto &device : device_list) {
        dlist.push_back(device);
        std::cout << "        -> " << device.get_info<sycl::info::device::name>()
                  << std::endl;
        //   auto queue = sycl::queue(device);
      }
      ++i;
    }

    if (gpu_dev == 999) {
      try {
        auto sel = sycl::gpu_selector();
        sel_dev = sel.select_device();
      } catch (...) {
        cout << "no gpu device found\n";
      }
      // sycl::default_selector ds;
      // sel_dev(ds);

    } else {
      if (gpu_dev > dlist.size() - 1) {
        cout << "ERROR: selected device index [" << gpu_dev << "] is too large\n";
        exit(1);
      }
      sel_dev = dlist[gpu_dev];
    }
    std::cout << "selected dev: "
              << sel_dev.get_info<sycl::info::device::name>()
              << "\n";

    if (showinfo) {
      cout << "\n";
      cout << "   name:    " << sel_dev.get_info<sycl::info::device::name>() 
           << "\n   vendor:  " << sel_dev.get_info<sycl::info::device::vendor>() 
           << "\n   version: " << sel_dev.get_info<sycl::info::device::version>() 
           // << "\n   OpenCL version: " << sel_dev.get_info<sycl::info::device::opencl_c_version>() 
           << "\n   profile: " << sel_dev.get_info<sycl::info::device::profile>()
           << "\n\n";
      cout << "   max compute units:   "
           << sel_dev.get_info<sycl::info::device::max_compute_units>()
           << endl;
      cout << "   max work item dim:   "
           << sel_dev.get_info<sycl::info::device::max_work_item_dimensions>()
           << endl;
      cout << "   max work item size:  "
           << sel_dev.get_info<sycl::info::device::max_work_item_sizes>().get(0)
           << " " <<  sel_dev.get_info<sycl::info::device::max_work_item_sizes>().get(1)
           << " "  << sel_dev.get_info<sycl::info::device::max_work_item_sizes>().get(2)
           << endl;
      cout << "   max work group size: "
           << sel_dev.get_info<sycl::info::device::max_work_group_size>()
           << endl;
      int mcf(0);
      cout << "   max clock frequency: ";
      try {
        mcf = sel_dev.get_info<sycl::info::device::max_clock_frequency>();
        cout << mcf << std::endl;
      } catch (...) {
        cout << "unsupported\n";
      }
      cout << "   max mem alloc:  " <<
        sel_dev.get_info<sycl::info::device::max_mem_alloc_size>() << endl;
      cout << "   max mem global: " << 
        sel_dev.get_info<sycl::info::device::global_mem_size>() << endl;
      cout << "   max mem local:  " << 
        sel_dev.get_info<sycl::info::device::local_mem_size>() << endl;
      exit(0);
    }
  }
  std::cout << std::endl;    

  auto exception_handler = [] (sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch(sycl::exception const& e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
        << e.what() << std::endl;
      }
    }
  };
  
  auto property_list =
    sycl::property_list{sycl::property::queue::enable_profiling()};  
  sycl::queue deviceQueue(sel_dev, exception_handler, property_list);


  std::cout << "do CPU: " << std::boolalpha << do_cpu << "\n";
  if (do_cpu) {
    std::cout << "   max prime: " << iter_cpu << "\n";
  }
  std::cout << "\n";

  vector<work> vwork;

  std::chrono::duration<double> diff {0};
  
  if (do_gpu) {

    for (int i=0; i<nthreadsGPU; ++i) {
      work w;
      w.id = i;
      w.size = N;
      w.niter = iter_gpu;      
      w.nitems = nitems;
      w.VRI.resize(N);
      w.success = false;
      w.queueLock = queueLock;
      if (sharedQueue) {
        w.deviceQueue = &deviceQueue;
      } else {
        w.deviceQueue = new sycl::queue( deviceQueue );
      }
    
      vwork.push_back( w );
    }
    
    
  }

  Crunch cr;

  if (do_cpu) {
    std::cout << "warming up CPU\n";
    auto swc = std::chrono::high_resolution_clock::now();
    cr.crunch(int(iter_cpu/2));
    auto ewc = std::chrono::high_resolution_clock::now();
    diff = ewc-swc;
    std::cout << "  done: " << diff.count() << " s\n";
  }

  std::vector<thread> tv,tc;
  tv.reserve(nthreadsGPU);
  tc.reserve(nthreadsCPU);

  auto start = std::chrono::high_resolution_clock::now();

  if (do_gpu) {
    for (int i=0; i<nthreadsGPU; ++i) {
      tv.push_back( thread{gpu_tsk(vwork[i])} );
    }
  }

  if (do_cpu) {
    for (int i=0; i<nthreadsCPU; ++i) {
      tc.push_back( thread{cpu_tsk(iter_cpu,cr)} );
    }
  }

  if (do_gpu) {
    for( auto &v: tv) {
      v.join();
    }
  }

  if (do_cpu) {
    for( auto &v: tc) {
      v.join();
    }
  }

  auto stop = std::chrono::high_resolution_clock::now();

  //  cout << "start: " << start.time_since_epoch().count() << endl;  
  
  float avgtime{0.};
  if (do_gpu) {
    
    std::sort(vwork.begin(), vwork.end(), sortBySubmitTime);

    auto mint = vwork[0].submit_time;
    auto minstart = vwork[0].start;

    
    for (auto &w : vwork ) {
      if (w.submit_time < mint) {
        mint = w.submit_time;
      }
      if (w.start < minstart) {
        minstart = w.start;
      }
    }      


    printf("%26s %4s   %4s   %4s   %4s\n"," ","subm","strt","end","run");
    
    for (auto &w : vwork) {
      int n=0;
      for (size_t i=2; i<w.size; ++i) {
        if (w.VRI[i]) { n++; }
      }
      avgtime += w.run;

      std::chrono::duration<double> d1 = w.start - minstart;
      std::chrono::duration<double> d2 = w.stop - minstart;
      std::chrono::duration<double> d3 = w.stop - w.start;
      
      std::cout << "GPU[" << w.id << "]:  nPrimes: " << n << "  "
                << std::fixed << std::setprecision(2)
                << std::setw(6)
                << float(w.submit_time-mint)/1000000000. << " "
                << std::setw(6) << float(w.start_time-mint) /1000000000. << " "
                << std::setw(6) << float(w.end_time-mint)   /1000000000. << " "
                << std::setw(6) << float(w.end_time-w.start_time)/1000000000.
                << "    "
                << std::setw(6) << d1.count() << " "
                << std::setw(6) << d2.count() << " "
                << std::setw(6) << d3.count()        
                << std::endl;

      if (!sharedQueue) delete w.deviceQueue;
      
    }
  }

  if (do_cpu) {
    std::cout << "CPU average time: " << cpusum/nthreadsCPU << std::endl;
  }
  if (do_gpu) {
    std::cout << "GPU average time: " << avgtime / vwork.size() << std::endl;
  }  

  diff = (stop - start);

  std::cout << "total time: " << diff.count() << std::endl;
    
  return 0;
  
}
