package org.apache.sysds.runtime.matrix.data;

public class ComplexDouble {
    public double re;
    public double im;

    public ComplexDouble(double re, double im){
        this.re = re;
        this.im = im;
    }

    public ComplexDouble add(ComplexDouble other){
        return new ComplexDouble(this.re + other.re, this.im + other.im);
    }

    public ComplexDouble sub(ComplexDouble other){
        return new ComplexDouble(this.re - other.re, this.im - other.im);
    }
    public ComplexDouble mul(ComplexDouble other){
        return new ComplexDouble(this.re * other.re - this.im * other.im, this.im * other.re + this.re * other.im);
    }

    /**
     * Powering of a complex double.
     * First, the polar form is calculated and then De Moivre's theorem is applied.
     *
     * @param n exponent
     * @return the n-th power of the complex double
     */
    public ComplexDouble pow(int n){
        double dist = Math.sqrt(this.re * this.re + this.im * this.im);
        double angle = Math.atan2(this.im, this.re);
    
        return new ComplexDouble(Math.pow(dist, n) * Math.cos(n * angle),
                                 Math.pow(dist, n) * Math.sin(n * angle));
    }

    // Division
    public ComplexDouble div(ComplexDouble other) {
        double denominator = other.re * other.re + other.im * other.im;
        return new ComplexDouble((this.re * other.re + this.im * other.im) / denominator,
                                    (this.im * other.re - this.re * other.im) / denominator);
    }

    // Conjugate
    public ComplexDouble conjugate() {
        return new ComplexDouble(this.re, -this.im);
    }

    // Absolute Value
    public double abs() {
        return Math.sqrt(this.re * this.re + this.im * this.im);
    }

    // Argument (Phase)
    public double arg() {
        return Math.atan2(this.im, this.re);
    }

    // Polar Form Conversion
    public static ComplexDouble fromPolar(double magnitude, double angle) {
        return new ComplexDouble(magnitude * Math.cos(angle), magnitude * Math.sin(angle));
    }


    @Override
    public String toString() {
        return this.re + " + " + this.im + "i";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ComplexDouble that = (ComplexDouble) o;

        double epsilon = 0.000001d;
        return Math.abs(this.re - that.re) < epsilon && Math.abs(this.im - that.im) < epsilon;
    }

}
