package org.apache.sysds.runtime.matrix.data;

import java.util.Arrays;

public class LibMatrixFourier {

    /**
     * Function to perform Fast Fourier Transformation on a given array.
     * Its length has to be a power of two.
     *
     * @param in array of doubles
     * @return array of ComplexDoubles
     */
    public static MatrixBlock[] fft(double[] in){

        int cols = in.length;
        MatrixBlock re = new MatrixBlock(1, cols, in);
        MatrixBlock im = new MatrixBlock(1, cols, new double[cols]);

        return fft_one_dim(re, im);
    }

    public static MatrixBlock[] fft(double[][] in){

        int cols = in[0].length;
        int rows = in.length;

        double[] flattened = Arrays.stream(in).flatMapToDouble(Arrays::stream).toArray();
        MatrixBlock re = new MatrixBlock(rows, cols, flattened);

        return fft(re);
    }

    public static MatrixBlock[] fft(MatrixBlock re){

        int rows = re.getNumRows();
        int cols = re.getNumColumns();

        if(!isPowerOfTwo(cols)) throw new RuntimeException("dimension is not power of two");

        MatrixBlock[][] res_rows = new MatrixBlock[rows][2];

        for(int i = 0; i < rows; i++){
            // use fft on each row
            double[] row_values = Arrays.copyOfRange(re.getDenseBlockValues(), i * cols, (i+1) * cols);
            res_rows[i] = fft_one_dim(new MatrixBlock(1, cols, row_values), new MatrixBlock(1, cols, new double[cols]));
        }

        if(rows == 1) return res_rows[0];

        double[][] res = new double[2][rows*cols];

        // flatten res_row
        for(int i = 0; i < rows; i++){
            double[] res_rows_re = res_rows[i][0].getDenseBlockValues();
            double[] res_rows_im = res_rows[i][1].getDenseBlockValues();
            for(int j = 0; j < cols; j++){
                res[0][i*cols+j] = res_rows_re[j];
                res[1][i*cols+j] = res_rows_im[j];
            }
        }

        for(int j = 0; j < cols; j++) {
            // double[re/im][] col_values
            double[][] col_values = new double[2][rows];
            for (int i = 0; i < rows; i++) {
                col_values[0][i] = res[0][i*cols+j];
                col_values[1][i] = res[1][i*cols+j];
            }

            MatrixBlock[] res_col = fft_one_dim(new MatrixBlock(1, rows, col_values[0]), new MatrixBlock(1, rows, col_values[1]));
            for (int i = 0; i < rows; i++) {
                res[0][i*cols+j] = res_col[0].getDenseBlockValues()[i];
                res[1][i*cols+j] = res_col[1].getDenseBlockValues()[i];
            }
        }

        return new MatrixBlock[]{new MatrixBlock(rows, cols, res[0]), new MatrixBlock(rows, cols, res[1])};

    }

    private static MatrixBlock[] fft_one_dim(MatrixBlock re, MatrixBlock im){

        int rows = re.getNumRows();
        int cols = re.getNumColumns();
        if(rows != 1) throw new RuntimeException("only for one dimension");

        if(cols == 1){
            return new MatrixBlock[]{re, im};
        }

        // 1st row real part, 2nd row imaginary part
        double[][] values = {re.getDenseBlockValues(), im.getDenseBlockValues()};

        // split values depending on index
        double[][] even = new double[2][cols/2];
        double[][] odd = new double[2][cols/2];
        for(int j = 0; j < cols/2; j++){
            for(int i = 0; i < 2; i++){
                even[i][j] = values[i][j*2];
                odd[i][j] = values[i][j*2+1];
            }
        }

        MatrixBlock[] res_even = fft_one_dim(new MatrixBlock(1, cols/2, even[0]), new MatrixBlock(1, cols/2, even[1]));
        MatrixBlock[] res_odd = fft_one_dim(new MatrixBlock(1, cols/2, odd[0]), new MatrixBlock(1, cols/2, odd[1]));

        double[][] res_even_values = new double[][]{
                res_even[0].getDenseBlockValues(),
                res_even[1].getDenseBlockValues()};

        double[][] res_odd_values = new double[][]{
                res_odd[0].getDenseBlockValues(),
                res_odd[1].getDenseBlockValues()};

        double angle = -2*Math.PI/cols;
        double[][] res = new double[2][cols];

        for(int j = 0; j < cols/2; j++){

            double[] omega_pow = new double[]{Math.cos(j*angle), Math.sin(j*angle)};

            // m = omega * res_odd[j]
            double[] m = new double[]{
                    omega_pow[0] * res_odd_values[0][j] - omega_pow[1] * res_odd_values[1][j],
                    omega_pow[0] * res_odd_values[1][j] + omega_pow[1] * res_odd_values[0][j]};

            // res[j] = res_even + m;
            // res[j+cols/2] = res_even - m;
            for(int i = 0; i < 2; i++){
                res[i][j] = res_even_values[i][j] + m[i];
                res[i][j+cols/2] = res_even_values[i][j] - m[i];
            }
        }

        return new MatrixBlock[]{new MatrixBlock(rows, cols, res[0]), new MatrixBlock(rows, cols, res[1])};
    }

    private static boolean isPowerOfTwo(int n){
        return ((n != 0) && ((n & (n - 1)) == 0)) || n == 1;
    }

    /**
     * Function to perform Fast Fourier Transformation on a given array.
     * Its length has to be a power of two.
     *
     * @param in array of ComplexDoubles
     * @return array of ComplexDoubles
     */
    public static ComplexDouble[] fft(ComplexDouble[] in){

        // TODO: how to invert fillToPowerOfTwo after calculation
        // in = fillToPowerOfTwo(in);

        int n = in.length;
        if(n == 1){
            return in;
        }

        double angle = -2*Math.PI/n;

        ComplexDouble[] even = new ComplexDouble[n/2];
        ComplexDouble[] odd = new ComplexDouble[n/2];
        for(int i=0; i < n/2; i++){
            even[i] = in[i*2];
            odd[i] = in[i*2+1];
        }
        ComplexDouble[] resEven = fft(even);
        ComplexDouble[] resOdd = fft(odd);

        ComplexDouble[] res = new ComplexDouble[n];
        for(int j=0; j < n/2; j++){
            ComplexDouble omegaPow = new ComplexDouble(Math.cos(j*angle), Math.sin(j*angle));

            res[j] = resEven[j].add(omegaPow.mul(resOdd[j]));
            res[j+n/2] = resEven[j].sub(omegaPow.mul(resOdd[j]));
        }
        
        return res;
    }

    /**
     * Function to perform Fast Fourier Transformation on a given array.
     * Its length has to be a power of two.
     *
     * @param in array of doubles
     * @return array of ComplexDoubles
     */
    public static ComplexDouble[] fft_old(double[] in){
        ComplexDouble[] complex = new ComplexDouble[in.length];
        for(int i=0; i<in.length; i++){
            complex[i] = new ComplexDouble(in[i],0);
        }
        return fft(complex);
    }

    public static MatrixBlock[] ifft_one_dim(MatrixBlock re, MatrixBlock im){

        int rows = re.getNumRows();
        int cols = re.getNumColumns();

        if(rows != 1) throw new RuntimeException("only for one dimension");

        double[] im_values = im.getDenseBlockValues();
        MatrixBlock conj = new MatrixBlock(rows, cols, Arrays.stream(im_values).map(i -> -i).toArray());

        MatrixBlock[] res = fft_one_dim(re, conj);

        double[] res_re = Arrays.stream(res[0].getDenseBlockValues()).map(i -> i/cols).toArray();
        double[] res_im = Arrays.stream(res[0].getDenseBlockValues()).map(i -> -i/cols).toArray();

        return new MatrixBlock[]{ new MatrixBlock(rows, cols, res_re), new MatrixBlock(rows, cols, res_im)};
    }

    public static ComplexDouble[] ifft(ComplexDouble[] in) {
        int n = in.length;
    
        // Conjugate the input array
        ComplexDouble[] conjugatedInput = new ComplexDouble[n];
        for (int i = 0; i < n; i++) {
            conjugatedInput[i] = in[i].conjugate();
        }
    
        // Apply FFT to conjugated input
        ComplexDouble[] fftResult = fft(conjugatedInput);
    
        // Conjugate the result of FFT and scale by n
        ComplexDouble[] ifftResult = new ComplexDouble[n];
        for (int i = 0; i < n; i++) {
            ifftResult[i] = new ComplexDouble(fftResult[i].re / n, -fftResult[i].im / n);
        }
    
        return ifftResult;
    }
    
    /**
     * IFFT for real-valued input.
     * @param in array of doubles
     * @return array of ComplexDoubles representing the IFFT
     */
    public static ComplexDouble[] ifft(double[] in) {
        ComplexDouble[] complexIn = new ComplexDouble[in.length];
        for (int i = 0; i < in.length; i++) {
            complexIn[i] = new ComplexDouble(in[i], 0);
        }
        return ifft(complexIn);
    }

    /**
     * Function to fill a given array of ComplexDoubles with 0-ComplexDoubles, so that the length is a power of two.
     * Needed for FastFourierTransformation
     *
     * @param in array of ComplexDoubles
     * @return array of ComplexDoubles
     */
    private static ComplexDouble[] fillToPowerOfTwo(ComplexDouble[] in){
        int missing = nextPowerOfTwo(in.length)- in.length;
        ComplexDouble[] res = new ComplexDouble[in.length+missing];
        for(int i=0; i<in.length; i++){
            res[i] = in[i];
        }
        for(int j=0; j<missing; j++){
            res[in.length+j] = new ComplexDouble(0,0);
        }
        return res;
    }

    /**
     * Function for calculating the next larger int which is a power of two
     *
     * @param n integer
     * @return next larger int which is a power of two
     */
    private static int nextPowerOfTwo(int n){
        int res = 1;
        while (res < n){
            res = res << 1;
        }
        return res;
    }

    /**
     * Function to perform Fast Fourier Transformation in a 2-dimensional array.
     * Both dimensions have to be a power of two.
     * First fft is applied to each row, then fft is applied to each column of the previous result.
     *
     * @param in 2-dimensional array of ComplexDoubles
     * @return 2-dimensional array of ComplexDoubles
     */
    public static ComplexDouble[][] fft2d(ComplexDouble[][] in) {

        int rows = in.length;
        int cols = in[0].length;

        ComplexDouble[][] out = new ComplexDouble[rows][cols];

        for(int i = 0; i < rows; i++){
            // use fft on row
            out[i] = fft(in[i]);
        }

        for(int j = 0; j < cols; j++){
            // get col as array
            ComplexDouble[] inCol = new ComplexDouble[rows];
            for(int i = 0; i < rows; i++){
                inCol[i] = out[i][j];
            }
            // use fft on col
            ComplexDouble[] resCol = fft(inCol);
            for (int i = 0; i < rows; i++) {
                out[i][j] = resCol[i];
            }
        }

        return out;
    }

    /**
     * Function to perform Fast Fourier Transformation in a 2-dimensional array.
     * Both dimensions have to be a power of two.
     * First fft is applied to each row, then fft is applied to each column of the previous result.
     *
     * @param in 2-dimensional array of doubles
     * @return 2-dimensional array of ComplexDoubles
     */
    public static ComplexDouble[][] fft2d(double[][] in){
        int rows = in.length;
        int cols = in[0].length;

        ComplexDouble[][] complex = new ComplexDouble[rows][cols];
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                complex[i][j] = new ComplexDouble(in[i][j],0);
            }
        }
        return fft2d(complex);
    }

       /**
     * Function to perform Inverse Fast Fourier Transformation in a 2-dimensional array.
     * Both dimensions have to be a power of two.
     * First ifft is applied to each row, then ifft is applied to each column of the previous result.
     *
     * @param in 2-dimensional array of ComplexDoubles
     * @return 2-dimensional array of ComplexDoubles
     */
    public static ComplexDouble[][] ifft2d(ComplexDouble[][] in) {
        int rows = in.length;
        int cols = in[0].length;

        ComplexDouble[][] out = new ComplexDouble[rows][cols];

        // Apply IFFT to each row
        for (int i = 0; i < rows; i++) {
            out[i] = ifft(in[i]);
        }

        // Apply IFFT to each column
        for (int j = 0; j < cols; j++) {
            ComplexDouble[] col = new ComplexDouble[rows];
            for (int i = 0; i < rows; i++) {
                col[i] = out[i][j];
            }

            ComplexDouble[] resCol = ifft(col);
            for (int i = 0; i < rows; i++) {
                out[i][j] = resCol[i];
            }
        }

        return out;
    }


}
