package org.apache.sysds.runtime.matrix.data;

public class LibMatrixFourier {

    /**
     * Function to perform Fast Fourier Transformation on a given array.
     * Its length has to be a power of two.
     *
     * @param in array of ComplexDoubles
     * @return array of ComplexDoubles
     */
    public static ComplexDouble[] fft(ComplexDouble[] in){

        // TODO: how to invert fillToPowerOfTwo after calculation
        //  in = fillToPowerOfTwo(in);

        int n = in.length;
        if(n == 1){
            return in;
        }

        double angle = 2*Math.PI/n;
        ComplexDouble omega = new ComplexDouble(Math.cos(angle), Math.sin(angle));

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
            res[j] = resEven[j].add(omega.pow(j).mul(resOdd[j]));
            res[j+n/2] = resEven[j].sub(omega.pow(j).mul(resOdd[j]));
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
    public static ComplexDouble[] fft(double[] in){
        ComplexDouble[] complex = new ComplexDouble[in.length];
        for(int i=0; i<in.length; i++){
            complex[i] = new ComplexDouble(in[i],0);
        }
        return fft(complex);
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

        // TODO: Is it sufficient to use only real parts of the result for further computations?
        int rows = in.length;
        int cols = in[0].length;

        ComplexDouble[][] out = new ComplexDouble[rows][cols];

        for(int i = 0; i < rows; i++){
            // use fft on row
            ComplexDouble[] resRow = fft(in[i]);
            // only use real part of result for further computations
            for(int j = 0; j < cols; j++){
                out[i][j] = resRow[j];
                //out[i][j] = new ComplexDouble(resRow[j].re, 0);
            }
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
                //out[i][j] = new ComplexDouble(resCol[i].re, 0);
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


}
