package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.ComplexDouble;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class FourierTest {

    @Test
    public void simple_test_one_dim() {
        // 1st row real part, 2nd row imaginary part
        double[][] in = {{0, 18, -15, 3},{0, 0, 0, 0}};
        double[][] expected = {{6, 15, -36, 15},{0, -15, 0, 15}};

        double[][] res = fft_one_dim(in);
        for(double[] row : res){
            for (double elem : row){
                System.out.print(elem + " ");
            }
            System.out.println();
        }
        assertArrayEquals(expected[0], res[0], 0.0001);
        assertArrayEquals(expected[1], res[1], 0.0001);
    }

    @Test
    public void simple_test_two_dim() {
        // tested with numpy
        double[][][] in = {{{0, 18},{-15, 3}},{{0, 0},{0, 0}}};

        double[][][] expected = {{{6, -36},{30, 0}},{{0, 0},{0, 0}}};

        double[][][] res = fft(in, false);

        for(double[][] matrix : res){
            for(double[] row : matrix) {
                for (double elem : row) {
                    System.out.print(elem + " ");
                }
                System.out.println();
            }
            System.out.println();
        }

        for(int k = 0; k < 2 ; k++){
            for(int i = 0; i < res[0].length; i++) {
                assertArrayEquals(expected[k][i], res[k][i], 0.0001);
            }
        }
    }

    @Test
    public void simple_test_one_dim_ifft() {

        double[][] in = {{1, -2, 3, -4},{0, 0, 0, 0}};

        double[][] res_fft = fft_one_dim(in);
        double[][] res = ifft_one_dim(res_fft);

        assertArrayEquals(in[0], res[0], 0.0001);
        assertArrayEquals(in[1], res[1], 0.0001);
    }

    @Test
    public void matrix_block_one_dim_test(){

        double[] in = {0, 18, -15, 3};

        double[] expected_re = {6,15,-36,15};
        double[] expected_im = {0,-15,0,15};

        MatrixBlock[] res = fft(in);
        double[] res_re = res[0].getDenseBlockValues();
        double[] res_im = res[1].getDenseBlockValues();

        for(double elem : res_re){
            System.out.print(elem+" ");
        }
        System.out.println();
        for(double elem : res_im){
            System.out.print(elem+" ");
        }

        assertArrayEquals(expected_re, res_re, 0.0001);
        assertArrayEquals(expected_im, res_im, 0.0001);
    }
    @Test
    public void matrix_block_two_dim_test(){

        double[][][] in = {{{0, 18},{-15, 3}}};

        double[] flattened_expected_re = {6,-36, 30,0};
        double[] flattened_expected_im = {0,0,0,0};

        MatrixBlock[] res = fft(in);
        double[] res_re = res[0].getDenseBlockValues();
        double[] res_im = res[1].getDenseBlockValues();

        for(double elem : res_re){
            System.out.print(elem+" ");
        }
        System.out.println();
        for(double elem : res_im){
            System.out.print(elem+" ");
        }

        assertArrayEquals(flattened_expected_re, res_re, 0.0001);
        assertArrayEquals(flattened_expected_im, res_im, 0.0001);
    }

}