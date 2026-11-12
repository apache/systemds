int main() {
    int n, total = 0, i = 1;
    scanf("%d", &n);
    while (i <= n) {
        total += i;
        i++;
    }
    printf("%d\n", total);
    return 0;
}