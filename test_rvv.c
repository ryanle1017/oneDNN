#include <riscv_vector.h>
#include <stdio.h>

int main() {
    size_t vl = __riscv_vsetvlmax_e32m2();
    printf("VLEN with m2: %zu\n", vl);
    return 0;
}
