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
#include "mkl.h"

using namespace std;

void printImage(const double* image, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << image[i * cols + j] << " ";
        }
        cout << endl;
    }
}

void m_img_transform(const double* img_in, int orig_w, int orig_h, int out_w, int out_h, double a, double b, double c, double d,
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
            1, 0, static_cast<double>(-cols / 2),
            0, 1, static_cast<double>(-rows / 2),
            0, 0, 1
    };
    double* t1 = t1_data;

    // Translation matrix for moving the origin back to the top left corner
    double t2_data[] = {
            1, 0, static_cast<double>(cols / 2),
            0, 1, static_cast<double>(rows / 2),
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
    int matrix_size = std::max(rows, cols);
    double m_data[matrix_size * matrix_size];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, t2, 3, rot, 3, 0.0, m_data, 3);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, m_data, 3, t1, 3, 0.0, m_data, 3);
    double* m = m_data;


    // Transform image
    //imgTransform(img_in, rows, cols, m[0], m[1], m[2], m[3], m[4], m[5], fill_value, img_out);
    m_img_transform(img_in,rows,cols,rows,cols,m[0], m[1], m[2], m[3], m[4], m[5], fill_value, img_out);
}
