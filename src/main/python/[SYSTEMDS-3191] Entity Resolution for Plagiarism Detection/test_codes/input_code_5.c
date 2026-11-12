int calc_factorial(int n) {
    if (n < 0) {
        // Factorial not defined for negative numbers
        return -1; 
    } else if (n == 0 || n == 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}


