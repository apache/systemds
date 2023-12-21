package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.math3.util.FastMath;
import java.util.Arrays;

public class LibMatrixFourier {

    /**
     * Function to perform Fast Fourier Transformation
     */

    public static MatrixBlock[] fft(MatrixBlock re, MatrixBlock im){

        int rows = re.getNumRows();
        int cols = re.getNumColumns();

        double[][][] in = new double[2][rows][cols];
        in[0] = convertToArray(re);
        in[1] = convertToArray(im);

        double[][][] res = fft(in, false);

        return convertToMatrixBlocks(res);
    }

    public static MatrixBlock[] ifft(MatrixBlock re, MatrixBlock im){

        int rows = re.getNumRows();
        int cols = re.getNumColumns();

        double[][][] in = new double[2][rows][cols];
        in[0] = convertToArray(re);
        in[1] = convertToArray(im);

        double[][][] res = fft(in, true);

        return convertToMatrixBlocks(res);
    }

    public static double[][][] fft(double[][][] in, boolean calcInv){

        int rows = in[0].length;
        int cols = in[0][0].length;

        double[][][] res = new double[2][rows][cols];

        for(int i = 0; i < rows; i++){
            // use fft or ifft on each row
            double[][] res_row;
            if(calcInv){
                res_row = ifft_one_dim(get_complex_row(in, i));
            } else {
                res_row = fft_one_dim(get_complex_row(in, i));
            }
            // set res row
            for (int j = 0; j < cols; j++){
                for( int k = 0; k < 2; k++){
                    res[k][i][j] = res_row[k][j];
                }
            }
        }

        if(rows == 1) return res;

        for(int j = 0; j < cols; j++){
            // use fft on each col
            double[][] res_col;
            if(calcInv){
                res_col = ifft_one_dim(get_complex_col(res, j));
            } else {
                res_col = fft_one_dim(get_complex_col(res, j));
            }
            // set res col
            for (int i = 0; i < rows; i++){
                for( int k = 0; k < 2; k++){
                    res[k][i][j] = res_col[k][i];
                }
            }
        }

        return res;
    }

    public static double[][] fft_one_dim(double[][] in){
        // 1st row real part, 2nd row imaginary part
        if(in == null || in.length != 2 || in[0].length != in[1].length) throw new RuntimeException("in false dimensions");

        int cols = in[0].length;
        if(cols == 1) return in;

        double angle = -2*FastMath.PI/cols;

        // split values depending on index
        double[][] even = new double[2][cols/2];
        double[][] odd = new double[2][cols/2];

        for(int i = 0; i < 2; i++){
            for (int j = 0; j < cols/2; j++){
                even[i][j] = in[i][j*2];
                odd[i][j] = in[i][j*2+1];
            }
        }
        double[][] res_even = fft_one_dim(even);
        double[][] res_odd = fft_one_dim(odd);

        double[][] res = new double[2][cols];

        for(int j=0; j < cols/2; j++){
            double[] omega_pow = new double[]{FastMath.cos(j*angle), FastMath.sin(j*angle)};

            // m = omega * res_odd[j]
            double[] m = new double[]{
                    omega_pow[0] * res_odd[0][j] - omega_pow[1] * res_odd[1][j],
                    omega_pow[0] * res_odd[1][j] + omega_pow[1] * res_odd[0][j]};

            // res[j] = res_even + m;
            // res[j+cols/2] = res_even - m;
            for(int i = 0; i < 2; i++){
                res[i][j] = res_even[i][j] + m[i];
                res[i][j+cols/2] = res_even[i][j] - m[i];
            }
        }

        return res;

    }

    public static double[][] ifft_one_dim(double[][] in) {

        // cols[0] is real part, cols[1] is imaginary part
        int cols = in[0].length;

        // conjugate input
        in[1] = Arrays.stream(in[1]).map(i -> -i).toArray();

        // apply fft
        double[][] res = fft_one_dim(in);

        // conjugate and scale result
        res[0] = Arrays.stream(res[0]).map(i -> i/cols).toArray();
        res[1] = Arrays.stream(res[1]).map(i -> -i/cols).toArray();

        return res;
    }

    private static MatrixBlock[] convertToMatrixBlocks(double[][][] in){

        int cols = in[0][0].length;
        int rows = in[0].length;

        double[] flattened_re = Arrays.stream(in[0]).flatMapToDouble(Arrays::stream).toArray();
        double[] flattened_im = new double[rows*cols];
        if(in.length > 1){
            flattened_im = Arrays.stream(in[1]).flatMapToDouble(Arrays::stream).toArray();
        }

        MatrixBlock re = new MatrixBlock(rows, cols, flattened_re);
        MatrixBlock im = new MatrixBlock(rows, cols, flattened_im);

        return new MatrixBlock[]{re, im};
    }

    private static MatrixBlock getZeroMatrixBlock(int rows, int cols){

        return new MatrixBlock(rows, cols, new double[cols*rows]);

    }

    private static double[][] convertToArray(MatrixBlock in){

        int rows = in.getNumRows();
        int cols = in.getNumColumns();

        double[][] out = new double[rows][cols];
        for(int i = 0; i < rows; i++){
            out[i] = Arrays.copyOfRange(in.getDenseBlockValues(), i * cols, (i+1) * cols);
        }

        return out;
    }
    private static double[][][] convertToArray(MatrixBlock[] in){

        int rows = in[0].getNumRows();
        int cols = in[0].getNumColumns();

        double[][][] out = new double[2][rows][cols];
        for(int k = 0; k < 2; k++){
            for(int i = 0; i < rows; i++){
                out[k][i] = Arrays.copyOfRange(in[k].getDenseBlockValues(), i * cols, (i+1) * cols);
            }
        }

        return out;
    }

    public static double[][] get_complex_row(double[][][] in, int i){

        int cols = in[0][0].length;

        double[][] row = new double[2][cols];
        // get row
        for (int j = 0; j < cols; j++){
            for( int k = 0; k < 2; k++){
                row[k][j] = in[k][i][j];
            }
        }
        return row;
    }
    public static double[][] get_complex_col(double[][][] in, int j){

        int rows = in[0].length;

        double[][] col = new double[2][rows];
        // get row
        for (int i = 0; i < rows; i++){
            for( int k = 0; k < 2; k++){
                col[k][i] = in[k][i][j];
            }
        }
        return col;
    }

    private static boolean isPowerOfTwo(int n){
        return ((n != 0) && ((n & (n - 1)) == 0)) || n == 1;
    }

    public static MatrixBlock[] fft(double[] in){
        double[][][] arr = new double[2][1][in.length];
        arr[0][0] = in;
        return fft(convertToMatrixBlocks(arr));
    }

    public static MatrixBlock[] fft(double[][][] in){
        return fft(convertToMatrixBlocks(in));
    }

    public static MatrixBlock[] fft(MatrixBlock[] in){
        return fft(in[0], in[1]);
    }

    public static MatrixBlock[] fft(MatrixBlock re){
        return fft(re, getZeroMatrixBlock(re.getNumRows(), re.getNumColumns()));
    }
}
