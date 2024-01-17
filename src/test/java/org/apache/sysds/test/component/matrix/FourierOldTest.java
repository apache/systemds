package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

import static org.apache.sysds.runtime.matrix.data.LibMatrixFourierOld.fft_one_dim_old;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourierOld.ifft_one_dim_old;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourierOld.fft_old;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourierOld.ifft_old;
import static org.junit.Assert.assertArrayEquals;

public class FourierOldTest {

    @Test
    public void simple_test_one_dim() {
        // 1st row real part, 2nd row imaginary part
        double[][] in = {{0, 18, -15, 3},{0, 0, 0, 0}};
        double[][] expected = {{6, 15, -36, 15},{0, -15, 0, 15}};

        double[][] res = fft_one_dim_old(in);

        assertArrayEquals(expected[0], res[0], 0.0001);
        assertArrayEquals(expected[1], res[1], 0.0001);
    }

    @Test
    public void simple_test_two_dim() {
        // tested with numpy
        double[][][] in = {{{0, 18},{-15, 3}},{{0, 0},{0, 0}}};

        double[][][] expected = {{{6, -36},{30, 0}},{{0, 0},{0, 0}}};

        double[][][] res = fft_old(in, false);

        for(int k = 0; k < 2 ; k++){
            for(int i = 0; i < res[0].length; i++) {
                assertArrayEquals(expected[k][i], res[k][i], 0.0001);
            }
        }
    }

    @Test
    public void simple_test_one_dim_ifft() {

        double[][] in = {{1, -2, 3, -4},{0, 0, 0, 0}};

        double[][] res_fft = fft_one_dim_old(in);
        double[][] res = ifft_one_dim_old(res_fft);

        assertArrayEquals(in[0], res[0], 0.0001);
        assertArrayEquals(in[1], res[1], 0.0001);
    }

    @Test
    public void matrix_block_one_dim_test(){

        double[] in = {0, 18, -15, 3};

        double[] expected_re = {6,15,-36,15};
        double[] expected_im = {0,-15,0,15};

        MatrixBlock[] res = fft_old(in);
        double[] res_re = res[0].getDenseBlockValues();
        double[] res_im = res[1].getDenseBlockValues();

        assertArrayEquals(expected_re, res_re, 0.0001);
        assertArrayEquals(expected_im, res_im, 0.0001);
    }
    @Test
    public void matrix_block_two_dim_test(){

        double[][][] in = {{{0, 18},{-15, 3}}};

        double[] flattened_expected_re = {6,-36, 30,0};
        double[] flattened_expected_im = {0,0,0,0};

        MatrixBlock[] res = fft_old(in);
        double[] res_re = res[0].getDenseBlockValues();
        double[] res_im = res[1].getDenseBlockValues();

        assertArrayEquals(flattened_expected_re, res_re, 0.0001);
        assertArrayEquals(flattened_expected_im, res_im, 0.0001);
    }

    @Test
    public void test_ifft_two_dim_matrixBlock() {

        MatrixBlock re = new MatrixBlock(2, 2,  new double[]{6,-36, 30, 0});
        MatrixBlock im = new MatrixBlock(2, 2,  new double[]{0, 0, 0, 0});

        double[] expected_re = {0, 18, -15, 3};
        double[] expected_im = {0, 0, 0, 0};

        MatrixBlock[] res = ifft_old(re, im);

        assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
        assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

    }

}
