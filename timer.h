/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

// A simple timer class

#ifdef __CUDACC__

// use CUDA's high-resolution timers when possible
#include <cuda_runtime_api.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <string>

void safe_call(cudaError_t error, const std::string& message = "")
{
  if(error)
    throw thrust::system_error(error, thrust::cuda_category(), message);
}

struct timer
{
  cudaEvent_t accStart;
  double accTime;

  timer(void)
  {
    safe_call(cudaEventCreate(&accStart));
  }

  ~timer(void)
  {
    safe_call(cudaEventDestroy(accStart));
  }

  void start(void)
  {
    safe_call(cudaEventRecord(accStart, 0));
  }

  void end(void)
  {
    cudaEvent_t end;

    safe_call(cudaEventCreate(&end));
    safe_call(cudaEventRecord(end, 0));
    safe_call(cudaEventSynchronize(end));

    float ms_elapsed;
    safe_call(cudaEventElapsedTime(&ms_elapsed, accStart, end));

    accTime = ms_elapsed / 1e3;

    safe_call(cudaEventDestroy(end));
  }

  double epsilon(void)
  {
    return 0.5e-6;
  }

};

#else

// fallback to clock()
#include <ctime>

struct timer
{
  clock_t accStart;
  double accTime;

  timer(void)
  {
  }

  ~timer(void)
  {
  }

  void start(void)
  {
    accStart = clock();
  }

  void end(void)
  {
    clock_t end;

    end = clock();
    accTime = static_cast<double>(end - accStart) / static_cast<double>(CLOCKS_PER_SEC);
  }

  double epsilon(void)
  {
    return 1.0 / static_cast<double>(CLOCKS_PER_SEC);
  }
};

#endif

