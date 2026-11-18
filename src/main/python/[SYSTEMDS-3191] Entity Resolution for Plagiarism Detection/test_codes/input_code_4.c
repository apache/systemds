int sumNumbers(int start, int end) {
    int sum = 0;
    int i = start;
    while (i <= end) {
        sum += i;
        i++;
    }
    return sum;
}
