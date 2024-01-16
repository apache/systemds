package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;
//import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.*;
import static org.junit.Assert.assertArrayEquals;
import static org.apache.sysds.runtime.matrix.data.LibMatrixSTFT.*;


public class LibMatrixSTFTTest {

    @Test
    public void simple_test() {

        double[] signal = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        int windowSize = 4;
        int overlap = 2;
        double[][] stftResult = one_dim_stft(signal, windowSize, overlap);

        // 1st row real part, 2nd row imaginary part
        double[][] expected = {{6, -2, -2, -2, 14, -2, -2, -2, 22, -2, -2, -2, 30, -2, -2, -2, 38, -2, -2, -2, 46, -2, -2, -2, 54, -2, -2, -2},{0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2}};

        for(double[] row : stftResult){
            for (double elem : row){
                System.out.print(elem + " ");
            }
            System.out.println();
        }
        assertArrayEquals(expected[0], stftResult[0], 0.0001);
        assertArrayEquals(expected[1], stftResult[1], 0.0001);
    }

    @Test
    public void matrix_block_one_dim_test(){

        double[] in = {0, 18, -15, 3};

        double[] expected_re = {6,15,-36,15};
        double[] expected_im = {0,-15,0,15};

        MatrixBlock[] res = stft(in, 4, 0);
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
    public void matrix_block_one_dim_test2(){

        double[] in = {10, 5, -3, 8, 15, -6, 2, 0};

        double[] expected_re = {20.0, 13.0, -6.0, 13.0, 14.0, -18.0, 10.0, -18.0, 11.0, 13.0, 23.0, 13.0};
        double[] expected_im = {0.0, 3.0, 0.0, -3.0, 0.0, -14.0, 0.0, 14.0, 0.0, 6.0, 0.0, -6.0 };

        MatrixBlock[] res = stft(in, 4, 2);
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

    /*
    public static void main(String[] args) {


        // Generate the sinusoidal signal
        //double[] signal = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        double[] signal = {10, 5, -3, 8, 15, -6, 2, 0};


        // Define STFT parameters
        int frameSize = 4;
        int overlap = 2;

        // Perform the STFT
        double[][] stftResult = one_dim_stft(signal, frameSize, overlap);

        // tensorflow change arguments names it is calles step
        // Output some results for verification
        // also for 2d array
        System.out.println("STFT Result (a few samples):");
        int l = stftResult[0].length;
        for (int i = 0; i < l; i++) {
            System.out.println("Real = " + stftResult[0][i] + ", Imaginary = " + stftResult[1][i]);
        }
    }
     */
}