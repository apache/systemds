int main() {
    int num;
   

    if (num < 0) {
        printf("Error: Negative input is not allowed.\n");
        return 1;
    }

    float result = 1; 
    int counter = 1;

    while (counter <= num) {
        result *= counter;
        counter++;
    }

    printf("Factorial of %d is %lld\n", num, result);
    return 0;
}