package org.apache.sysds.runtime.matrix.data;

public class LibMatrixFourier2 {


    public static ComplexDouble[] fft(double[] in){
        ComplexDouble[] complex = new ComplexDouble[in.length];
        double[] im = new double[in.length];
        transform(in, im);
        for(int i=0; i<in.length; i++){
            complex[i] = new ComplexDouble(in[i], im[i]);
        }
        return complex;
    }
    /*
     * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
     * The vector can have any length. This is a wrapper function.
     */
    public static void transform(double[] real, double[] imag) {
        int n = real.length;
        if (n != imag.length)
            throw new IllegalArgumentException("Mismatched lengths");
        if (n == 0)
            return;
        else if ((n & (n - 1)) == 0)  // Is power of 2
            transformRadix2(real, imag);
        else  // More complicated algorithm for arbitrary sizes
            transformBluestein(real, imag);
    }


    /*
     * Computes the inverse discrete Fourier transform (IDFT) of the given complex vector, storing the result back into the vector.
     * The vector can have any length. This is a wrapper function. This transform does not perform scaling, so the inverse is not a true inverse.
     */
    public static void inverseTransform(double[] real, double[] imag) {
        transform(imag, real);
    }


    /*
     * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
     * The vector's length must be a power of 2. Uses the Cooley-Tukey decimation-in-time radix-2 algorithm.
     */
    public static void transformRadix2(double[] real, double[] imag) {
        // Length variables
        int n = real.length;
        if (n != imag.length)
            throw new IllegalArgumentException("Mismatched lengths");
        int levels = 31 - Integer.numberOfLeadingZeros(n);  // Equal to floor(log2(n))
        if (1 << levels != n)
            throw new IllegalArgumentException("Length is not a power of 2");

        // Trigonometric tables
        double[] cosTable = new double[n / 2];
        double[] sinTable = new double[n / 2];
        for (int i = 0; i < n / 2; i++) {
            cosTable[i] = Math.cos(2 * Math.PI * i / n);
            sinTable[i] = Math.sin(2 * Math.PI * i / n);
        }

        // Bit-reversed addressing permutation
        for (int i = 0; i < n; i++) {
            int j = Integer.reverse(i) >>> (32 - levels);
            if (j > i) {
                double temp = real[i];
                real[i] = real[j];
                real[j] = temp;
                temp = imag[i];
                imag[i] = imag[j];
                imag[j] = temp;
            }
        }

        // Cooley-Tukey decimation-in-time radix-2 FFT
        for (int size = 2; size <= n; size *= 2) {
            int halfsize = size / 2;
            int tablestep = n / size;
            for (int i = 0; i < n; i += size) {
                for (int j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
                    int l = j + halfsize;
                    double tpre =  real[l] * cosTable[k] + imag[l] * sinTable[k];
                    double tpim = -real[l] * sinTable[k] + imag[l] * cosTable[k];
                    real[l] = real[j] - tpre;
                    imag[l] = imag[j] - tpim;
                    real[j] += tpre;
                    imag[j] += tpim;
                }
            }
            if (size == n)  // Prevent overflow in 'size *= 2'
                break;
        }
    }


    /*
     * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
     * The vector can have any length. This requires the convolution function, which in turn requires the radix-2 FFT function.
     * Uses Bluestein's chirp z-transform algorithm.
     */
    public static void transformBluestein(double[] real, double[] imag) {
        // Find a power-of-2 convolution length m such that m >= n * 2 + 1
        int n = real.length;
        if (n != imag.length)
            throw new IllegalArgumentException("Mismatched lengths");
        if (n >= 0x20000000)
            throw new IllegalArgumentException("Array too large");
        int m = Integer.highestOneBit(n) * 4;

        // Trigonometric tables
        double[] cosTable = new double[n];
        double[] sinTable = new double[n];
        for (int i = 0; i < n; i++) {
            int j = (int)((long)i * i % (n * 2));  // This is more accurate than j = i * i
            cosTable[i] = Math.cos(Math.PI * j / n);
            sinTable[i] = Math.sin(Math.PI * j / n);
        }

        // Temporary vectors and preprocessing
        double[] areal = new double[m];
        double[] aimag = new double[m];
        for (int i = 0; i < n; i++) {
            areal[i] =  real[i] * cosTable[i] + imag[i] * sinTable[i];
            aimag[i] = -real[i] * sinTable[i] + imag[i] * cosTable[i];
        }
        double[] breal = new double[m];
        double[] bimag = new double[m];
        breal[0] = cosTable[0];
        bimag[0] = sinTable[0];
        for (int i = 1; i < n; i++) {
            breal[i] = breal[m - i] = cosTable[i];
            bimag[i] = bimag[m - i] = sinTable[i];
        }

        // Convolution
        double[] creal = new double[m];
        double[] cimag = new double[m];
        convolve(areal, aimag, breal, bimag, creal, cimag);

        // Postprocessing
        for (int i = 0; i < n; i++) {
            real[i] =  creal[i] * cosTable[i] + cimag[i] * sinTable[i];
            imag[i] = -creal[i] * sinTable[i] + cimag[i] * cosTable[i];
        }
    }


    /*
     * Computes the circular convolution of the given real vectors. Each vector's length must be the same.
     */
    public static void convolve(double[] xvec, double[] yvec, double[] outvec) {
        int n = xvec.length;
        if (n != yvec.length || n != outvec.length)
            throw new IllegalArgumentException("Mismatched lengths");
        convolve(xvec, new double[n], yvec, new double[n], outvec, new double[n]);
    }


    /*
     * Computes the circular convolution of the given complex vectors. Each vector's length must be the same.
     */
    public static void convolve(double[] xreal, double[] ximag,
                                double[] yreal, double[] yimag, double[] outreal, double[] outimag) {

        int n = xreal.length;
        if (n != ximag.length || n != yreal.length || n != yimag.length
                || n != outreal.length || n != outimag.length)
            throw new IllegalArgumentException("Mismatched lengths");

        xreal = xreal.clone();
        ximag = ximag.clone();
        yreal = yreal.clone();
        yimag = yimag.clone();
        transform(xreal, ximag);
        transform(yreal, yimag);

        for (int i = 0; i < n; i++) {
            double temp = xreal[i] * yreal[i] - ximag[i] * yimag[i];
            ximag[i] = ximag[i] * yreal[i] + xreal[i] * yimag[i];
            xreal[i] = temp;
        }
        inverseTransform(xreal, ximag);

        for (int i = 0; i < n; i++) {  // Scaling (because this FFT implementation omits it)
            outreal[i] = xreal[i] / n;
            outimag[i] = ximag[i] / n;
        }
    }

}
