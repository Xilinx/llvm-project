// RUN: mlir-tosa-cpu-runner %s -e main --entry-point-result=void --shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext,%mlir_lib_dir/libmlir_test_tosa_cpu_runner_c_wrappers%shlibext

// CHECK: [8,  8,  8,  8,  8,  8]
module attributes {} {
  func.func @main() -> () {                        
    %const = "tosa.const"() {value = dense<1.000000e-07> : tensor<f32>} : () -> tensor<f32>
    %0 = "tosa.abs"(%const) : (tensor<f32>) -> tensor<f32>                         
    return
    }  
}
