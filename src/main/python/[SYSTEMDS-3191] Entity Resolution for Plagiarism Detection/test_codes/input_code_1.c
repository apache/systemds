int addValues(int begin, int finish) {
    int total = 0;
    for (int idx = begin; idx <= finish; idx++) {
        total += idx;
    }
    return total;
}