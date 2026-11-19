/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


#ifndef CUJAVA_CUBLAS_COMMON_HPP
#define CUJAVA_CUBLAS_COMMON_HPP

#include <jni.h>
#include <cublas_v2.h>      // cuBLAS v1 is deprecated
#include <cuda_runtime.h>

#include "../common/cujava_logger.hpp"
#include "../common/cujava_jni_utils.hpp"
#include "../common/cujava_pointer_utils.hpp"

#define CUJAVA_CUBLAS_INTERNAL_ERROR (-1)

#endif // CUJAVA_CUBLAS_COMMON_HPP

