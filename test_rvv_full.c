#include <riscv_vector.h>
#include <stdio.h>

int main() {
    printf("Testing RVV intrinsics:\n");
    
    size_t vl_m1 = __riscv_vsetvlmax_e32m1();
    size_t vl_m2 = __riscv_vsetvlmax_e32m2();
    size_t vl_m4 = __riscv_vsetvlmax_e32m4();
    
    printf("VLEN with m1: %zu\n", vl_m1);
    printf("VLEN with m2: %zu\n", vl_m2);
    printf("VLEN with m4: %zu\n", vl_m4);
    
    // Test basic operations
    float data[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float result[16];
    
    size_t vl = __riscv_vsetvl_e32m2(16);
    vfloat32m2_t v = __riscv_vle32_v_f32m2(data, vl);
    v = __riscv_vfmul_vf_f32m2(v, 2.0f, vl);
    __riscv_vse32_v_f32m2(result, v, vl);
    
    printf("Multiplication test: %.1f * 2 = %.1f\n", data[0], result[0]);
    
    return 0;
}
