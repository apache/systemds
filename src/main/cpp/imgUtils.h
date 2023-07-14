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

#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

void printImage(const double* image, int rows, int cols);

void m_img_transform(const double* img_in, int orig_w, int orig_h, int out_w, int out_h, double a, double b, double c, double d,
                     double e, double f, double fill_value, double* img_out);

void imageRotate(double* img_in, int rows, int cols, double radians, double fill_value, double* img_out);

#endif /* IMAGE_UTILS_H */
