package org.apache.sysds.runtime.matrix.data;

public class ComplexDouble {
    double re;
    double im;

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
        // compute polar form
        double dist = Math.sqrt(Math.pow(this.re,2) + Math.pow(this.im,2));
        double angle = Math.acos(this.re/dist);

        // de moivre’s theorem
        return new ComplexDouble(Math.pow(dist,n) * Math.cos(n*angle),Math.pow(dist,n) * Math.sin(n*angle));
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
