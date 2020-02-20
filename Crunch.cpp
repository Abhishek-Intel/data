#include "Crunch.h"
#include <chrono>

unsigned int Crunch::crunch( const unsigned int n_iterations ) const {

  std::chrono::duration<double> diff {0};

  auto start = std::chrono::high_resolution_clock::now();
  // Flag to trigger the allocation
  bool is_prime;

  // Let's prepare the material for the allocations
  unsigned int   primes_size = 1;
  unsigned long* primes      = new unsigned long[primes_size];
  primes[0]                  = 2;

  unsigned long i = 2;

  // Loop on numbers
  for ( unsigned long int iiter = 0; iiter < n_iterations; iiter++ ) {
    // Once at max, it returns to 0
    i += 1;

    // Check if it can be divided by the smaller ones
    is_prime = true;
    for ( unsigned long j = 2; j < i && is_prime; ++j ) {
      if ( i % j == 0 ) is_prime = false;
    } // end loop on numbers < than tested one

    if ( is_prime ) {
      // copy the array of primes (INEFFICIENT ON PURPOSE!)
      unsigned int   new_primes_size = 1 + primes_size;
      unsigned long* new_primes      = new unsigned long[new_primes_size];

      for ( unsigned int prime_index = 0; prime_index < primes_size; prime_index++ ) {
        new_primes[prime_index] = primes[prime_index];
      }
      // attach the last prime
      new_primes[primes_size] = i;

      // Update primes array
      delete[] primes;
      primes      = new_primes;
      primes_size = new_primes_size;
    } // end is prime

  } // end of while loop

  // Fool Compiler optimisations:
  for ( unsigned int prime_index = 0; prime_index < primes_size; prime_index++ )
    if ( primes[prime_index] == 4 )

  delete[] primes;

  auto end = std::chrono::high_resolution_clock::now();
  diff = end - start;
  return (diff.count()*1000000);

  
}
