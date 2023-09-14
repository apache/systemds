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

#include <iostream>
#include <cmath>
#include <cstring>
#include "common.h"

using namespace std;

void printImage(const double* image, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << image[i * cols + j] << " ";
        }
        cout <<  endl;
    }
    cout << "\n"<< endl;
}

void img_transform(const double* img_in, int orig_w, int orig_h, int out_w, int out_h, double a, double b, double c, double d,
                     double e, double f, double fill_value, double* img_out) {
    double divisor = a * e - b * d;
    if (divisor == 0) {
        std::cout << "Inverse matrix does not exist! Returning input." << std::endl;
        for (int i = 0; i < orig_h; i++) {
            for (int j = 0; j < orig_w; j++) {
                img_out[i * orig_w + j] = img_in[i * orig_w + j];
            }
        }
    } else {
        // Create the inverted transformation matrix
        double T_inv[9];
        T_inv[0] = e / divisor;
        T_inv[1] = -b / divisor;
        T_inv[2] = (b * f - c * e) / divisor;
        T_inv[3] = -d / divisor;
        T_inv[4] = a / divisor;
        T_inv[5] = (c * d - a * f) / divisor;
        T_inv[6] = 0.0;
        T_inv[7] = 0.0;
        T_inv[8] = 1.0;

        // Create the coordinates of output pixel centers linearized in row-major order
        double* coords = new double[2 * out_w * out_h];
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                coords[2 * (i * out_w + j)] = j + 0.5;
                coords[2 * (i * out_w + j) + 1] = i + 0.5;
            }
        }

        // Perform matrix multiplication to compute sampling pixel indices
        double* transformed_coords = new double[2 * out_w * out_h];
        for (int i = 0; i < out_w * out_h; i++) {
            double x = coords[2 * i];
            double y = coords[2 * i + 1];
            transformed_coords[2 * i] = std::floor(T_inv[0] * x + T_inv[1] * y + T_inv[2]) + 1;
            transformed_coords[2 * i + 1] = std::floor(T_inv[3] * x + T_inv[4] * y + T_inv[5]) + 1;
        }

        // Fill output image
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                int inx = static_cast<int>(transformed_coords[2 * (i * out_w + j)]) - 1;
                int iny = static_cast<int>(transformed_coords[2 * (i * out_w + j) + 1]) - 1;
                if (inx >= 0 && inx < orig_w && iny >= 0 && iny < orig_h) {
                    img_out[i * out_w + j] = img_in[iny * orig_w + inx];
                } else {
                    img_out[i * out_w + j] = fill_value;
                }
            }
        }

        delete[] coords;
        delete[] transformed_coords;
    }
}

void imageRotate(double* img_in, int rows, int cols, double radians, double fill_value, double* img_out) {
    // Translation matrix for moving the origin to the center of the image
    double t1_data[] = {
            1, 0, static_cast<double>(-cols)/2,
            0, 1, static_cast<double>(-rows)/2,
            0, 0, 1
    };
    double* t1 = t1_data;
    // Translation matrix for moving the origin back to the top left corner
    double t2_data[] = {
            1, 0, static_cast<double>(cols)/2,
            0, 1, static_cast<double>(rows)/2,
            0, 0, 1
    };
    double* t2 = t2_data;
    // The rotation matrix around the origin
    double rot_data[] = {
            cos(radians), sin(radians), 0,
            -sin(radians), cos(radians), 0,
            0, 0, 1
    };
    double* rot = rot_data;

    // Combined transformation matrix
    double m_data1[3*3];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, t2, 3, rot, 3, 0.0, m_data1, 3);
    double m_data2[3*3];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, m_data1, 3, t1, 3, 0.0, m_data2, 3);
    double* m = m_data2;
    // Transform image
    img_transform(img_in,rows,cols,rows,cols,m[0], m[1], m[2], m[3], m[4], m[5], fill_value, img_out);
}

double* imageCutout(double* img_in, int rows, int cols, int x, int y, int width, int height, double fill_value) {
    // Allocate memory for the output image using MKL
   double* img_out = new double[rows * cols];

    if (width < 1 || height < 1) {
        // Invalid width or height, return the input image as it is
        cblas_dcopy(rows * cols, img_in, 1, img_out, 1);
    } else {
        int end_x = x + width - 1;
        int end_y = y + height - 1;

        int start_x = std::max(1, x);
        int start_y = std::max(1, y);
        end_x = std::min(cols, end_x);
        end_y = std::min(rows, end_y);

        // Copy the input image to the output image using MKL
        cblas_dcopy(rows * cols, img_in, 1, img_out, 1);

        // Fill the cutout region with the fill_value
        for (int i = start_y - 1; i < end_y; ++i) {
            for (int j = start_x - 1; j < end_x; ++j) {
                img_out[i * cols + j] = fill_value;
            }
        }
    }

    return img_out;
}

double* imageCrop(double* img_in, int orig_w, int orig_h, int w, int h, int x_offset, int y_offset) {
    // Allocate memory for the output image
    double* img_out = new double[w * h];

    int start_h = (std::ceil((orig_h - h) / 2)) + y_offset - 1 ;
    int end_h = (start_h + h - 1);
    int start_w = (std::ceil((orig_w - w) / 2)) + x_offset - 1;
    int end_w = (start_w + w - 1);

    // Create a mask to identify the cropped region
    double* mask = new double[orig_w * orig_h];
    double* temp_mask = new double[w * h];

    // Set mask elements to 0 outside the cropped region and 1 inside
    memset(mask, 0, orig_w * orig_h * sizeof(double));
    for(int i = 0; i < h * w; i++) {
     temp_mask[i] = 1;
    }

    for (int i = start_h; i <= end_h; ++i) {
        for (int j = start_w; j <= end_w; ++j) {
            mask[i * orig_w + j] = temp_mask[(i - start_h) * w + (j - start_w)];
        }
    }

    // Apply the mask to crop the image
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            img_out[i * w + j] = img_in[(start_h + i) * orig_w + (start_w + j)] * mask[(start_h + i) * orig_w + (start_w + j)];
        }
    }

    // Free memory for the mask
    delete[] mask;
    delete[] temp_mask;

    return img_out;
}

void img_translate(double* img_in, double offset_x, double offset_y,
                   int in_w, int in_h, int out_w, int out_h, double fill_value, double* img_out) {
    int w = out_w;
    int h = out_h;

    offset_x = round(offset_x);
    offset_y = round(offset_y);


    int start_x = 0 - static_cast<int>(offset_x);
    int start_y = 0 - static_cast<int>(offset_y);
    int end_x = std::max(w, out_w) - static_cast<int>(offset_x);
    int end_y = std::max(h, out_h) - static_cast<int>(offset_y);

    if (start_x < 0)
        start_x = 0;
    if (start_y < 0)
        start_y = 0;

    if (w < end_x)
        end_x = w;
    if (h < end_y)
        end_y = h;

    if (out_w < end_x + static_cast<int>(offset_x))
        end_x = out_w - static_cast<int>(offset_x);
    if (out_h < end_y + static_cast<int>(offset_y))
        end_y = out_h - static_cast<int>(offset_y);

    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            img_out[y * out_w + x] = fill_value;
        }
    }

   if (start_x < end_x && start_y < end_y) {
           for (int y = start_y + static_cast<int>(offset_y); y < end_y + static_cast<int>(offset_y); ++y) {
                int x_in = (start_x > 0) ? start_x + static_cast<int>(offset_x): start_x;
                int y_in = (start_y > 0 ) ? y : y - static_cast<int>(offset_y);
               cblas_dcopy(end_x - start_x, &img_in[(x_in) + (y_in) * in_w],
                           1, &img_out[y * out_w + start_x + static_cast<int>(offset_x)], 1);
           }
       }
}

