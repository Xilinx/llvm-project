// RUN: mlir-opt %s -split-input-file -linalg-unfuse | FileCheck %s

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x10x10xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x8x8xf32>
func.func @unfuse_conv_2d_tensor_add(%ifm : tensor<1x1024x10x10xf32>, %summand : tensor<1x1024x8x8xf32>) -> tensor<1x1024x8x8xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>

    %init = tensor.splat %zero : tensor<1x1024x8x8xf32>
    %result = linalg.conv_2d_tensor_add
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %weights, %bias, %summand : tensor<1x1024x10x10xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, tensor<1x1024x8x8xf32>)
        outs(%init : tensor<1x1024x8x8xf32>)
        -> tensor<1x1024x8x8xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :
    // CHECK: %[[out:.+]] = arith.addf %[[conv]], %[[summand]]

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x8x8xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add_globalaveragepool(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x10x10xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x8x8xf32>
func.func @unfuse_conv_2d_tensor_add_globalaveragepool(%ifm : tensor<1x1024x10x10xf32>, %summand : tensor<1x1024x8x8xf32>) -> tensor<1x1024x1x1xf32> {
    // CHECK-DAG: %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1024x1x1xf32>
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[cst0:.+]] = arith.constant dense<6.400000e+01> : tensor<1x1024x1x1xf32>

    %init = tensor.splat %zero : tensor<1x1024x1x1xf32>
    %result = linalg.conv_2d_tensor_add_globalaveragepool
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %weights, %bias, %summand : tensor<1x1024x10x10xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, tensor<1x1024x8x8xf32>)
        outs(%init : tensor<1x1024x1x1xf32>)
        -> tensor<1x1024x1x1xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :
    // CHECK: %[[add:.+]] = arith.addf %[[conv]], %[[summand]]
    // CHECK: %[[init:.+]] = tensor.empty()
    // CHECK: %[[pool:.+]] = linalg.pooling_nchw_sum
    // CHECK-SAME: ins(%[[add]], %[[init]] :
    // CHECK-SAME: outs(%[[cst]] :
    // CHECK: %[[out:.+]] = arith.divf %[[pool]], %[[cst0]]

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x1x1xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_relu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x17x17xf32>
func.func @unfuse_conv_2d_relu(%ifm : tensor<1x1024x17x17xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>

    %init = tensor.splat %zero : tensor<1x1024x7x7xf32>
    %result = linalg.conv_2d_relu
        {dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
        ins(%ifm, %weights, %bias : tensor<1x1024x17x17xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>)
        outs(%init : tensor<1x1024x7x7xf32>)
        -> tensor<1x1024x7x7xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw

    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :

    // CHECK: %[[out:.+]] = linalg.relu_2d_nchw
    // CHECK-SAME: ins(%[[conv]] :
    // CHECK-SAME: outs(%[[conv]] :

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x7x7xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add_relu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x17x17xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x7x7xf32>
func.func @unfuse_conv_2d_tensor_add_relu(%ifm : tensor<1x1024x17x17xf32>, %summand : tensor<1x1024x7x7xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>

    %init = tensor.splat %zero : tensor<1x1024x7x7xf32>
    %result = linalg.conv_2d_tensor_add_relu
        {dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
        ins(%ifm, %weights, %bias, %summand : tensor<1x1024x17x17xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, tensor<1x1024x7x7xf32>)
        outs(%init : tensor<1x1024x7x7xf32>)
        -> tensor<1x1024x7x7xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw

    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :

    // CHECK: %[[add:.+]] = arith.addf %[[conv]], %[[summand]]

    // CHECK: %[[out:.+]] = linalg.relu_2d_nchw
    // CHECK-SAME: ins(%[[add]] :
    // CHECK-SAME: outs(%[[add]] :

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x7x7xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add_relu_globalaveragepool(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x10x10xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x8x8xf32>
func.func @unfuse_conv_2d_tensor_add_relu_globalaveragepool(%ifm : tensor<1x1024x10x10xf32>, %summand : tensor<1x1024x8x8xf32>) -> tensor<1x1024x1x1xf32> {
    // CHECK-DAG: %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1024x1x1xf32>
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[cst0:.+]] = arith.constant dense<6.400000e+01> : tensor<1x1024x1x1xf32>

    %init = tensor.splat %zero : tensor<1x1024x1x1xf32>
    %result = linalg.conv_2d_tensor_add_relu_globalaveragepool
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %weights, %bias, %summand : tensor<1x1024x10x10xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, tensor<1x1024x8x8xf32>)
        outs(%init : tensor<1x1024x1x1xf32>)
        -> tensor<1x1024x1x1xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :
    // CHECK: %[[add:.+]] = arith.addf %[[conv]], %[[summand]]
    // CHECK: %[[relu:.+]] = linalg.relu_2d_nchw
    // CHECK-SAME: ins(%[[add]] :
    // CHECK-SAME: outs(%[[add]] :
    // CHECK: %[[init:.+]] = tensor.empty()
    // CHECK: %[[pool:.+]] = linalg.pooling_nchw_sum
    // CHECK-SAME: ins(%[[relu]], %[[init]] :
    // CHECK-SAME: outs(%[[cst]] :
    // CHECK: %[[out:.+]] = arith.divf %[[pool]], %[[cst0]]

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x1x1xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_lrelu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
func.func @unfuse_conv_2d_lrelu(%ifm : tensor<1x1024x15x15xf32>) -> tensor<1x1024x13x13xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[alpha:.+]] = arith.constant 2.000000e-02 : f32
    %alpha = arith.constant 2.000000e-02 : f32

    %init = tensor.splat %zero : tensor<1x1024x13x13xf32>
    %result = linalg.conv_2d_lrelu
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %weights, %bias, %alpha : tensor<1x1024x15x15xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, f32)
        outs(%init : tensor<1x1024x13x13xf32>)
        -> tensor<1x1024x13x13xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK-SAME: ins(%[[bias]] :

    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :

    // CHECK: %[[out:.+]] = linalg.lrelu_2d_nchw
    // CHECK-SAME: ins(%[[conv]], %[[alpha]] :
    // CHECK-SAME: outs(%[[conv]] :

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x13x13xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add_lrelu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x13x13xf32>
func.func @unfuse_conv_2d_tensor_add_lrelu(%ifm : tensor<1x1024x15x15xf32>, %summand : tensor<1x1024x13x13xf32>) -> tensor<1x1024x13x13xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[alpha:.+]] = arith.constant 2.000000e-02 : f32
    %alpha = arith.constant 2.000000e-02 : f32

    %init = tensor.splat %zero : tensor<1x1024x13x13xf32>
    %result = linalg.conv_2d_tensor_add_lrelu
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %weights, %bias, %summand, %alpha : tensor<1x1024x15x15xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, tensor<1x1024x13x13xf32>, f32)
        outs(%init : tensor<1x1024x13x13xf32>)
        -> tensor<1x1024x13x13xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK-SAME: ins(%[[bias]] :

    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :

    // CHECK: %[[add:.+]] = arith.addf %[[conv]], %[[summand]]

    // CHECK: %[[out:.+]] = linalg.lrelu_2d_nchw
    // CHECK-SAME: ins(%[[add]], %[[alpha]] :
    // CHECK-SAME: outs(%[[add]] :

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x13x13xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add_relu_globalaveragepool(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x10x10xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x8x8xf32>
func.func @unfuse_conv_2d_tensor_add_relu_globalaveragepool(%ifm : tensor<1x1024x10x10xf32>, %summand : tensor<1x1024x8x8xf32>) -> tensor<1x1024x1x1xf32> {
    // CHECK-DAG: %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1024x1x1xf32>
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[alpha:.+]] = arith.constant 2.000000e-02 : f32
    %alpha = arith.constant 2.000000e-02 : f32
    // CHECK-DAG: %[[cst0:.+]] = arith.constant dense<6.400000e+01> : tensor<1x1024x1x1xf32>

    %init = tensor.splat %zero : tensor<1x1024x1x1xf32>
    %result = linalg.conv_2d_tensor_add_lrelu_globalaveragepool
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %weights, %bias, %summand, %alpha : tensor<1x1024x10x10xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, tensor<1x1024x8x8xf32>, f32)
        outs(%init : tensor<1x1024x1x1xf32>)
        -> tensor<1x1024x1x1xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :
    // CHECK: %[[add:.+]] = arith.addf %[[conv]], %[[summand]]
    // CHECK: %[[lrelu:.+]] = linalg.lrelu_2d_nchw
    // CHECK-SAME: ins(%[[add]], %[[alpha]] :
    // CHECK-SAME: outs(%[[add]] :
    // CHECK: %[[init:.+]] = tensor.empty()
    // CHECK: %[[pool:.+]] = linalg.pooling_nchw_sum
    // CHECK-SAME: ins(%[[lrelu]], %[[init]] :
    // CHECK-SAME: outs(%[[cst]] :
    // CHECK: %[[out:.+]] = arith.divf %[[pool]], %[[cst0]]

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x1x1xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_lrelu_maxpool(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
func.func @unfuse_conv_2d_lrelu_maxpool(%ifm : tensor<1x1024x15x15xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[alpha:.+]] = arith.constant 2.000000e-02 : f32
    %alpha = arith.constant 2.000000e-02 : f32

    // CHECK-DAG: %[[pad_value:.+]] = arith.constant 0xFF800000 : f32

    %init = tensor.splat %zero : tensor<1x1024x7x7xf32>
    %result = linalg.conv_2d_lrelu_maxpool
        {
            dilations = dense<1> : tensor<2xi64>,
            strides = dense<1> : tensor<2xi64>,
            mpKernelSize = dense<2> : tensor<2xi64>,
            mpStrides = dense<2> : tensor<2xi64>,
            mpDilations = dense<1> : tensor<2xi64>,
            mpPadding = dense<[0, 1, 0, 1]> : tensor<4xi64>
        }
        ins(%ifm, %weights, %bias, %alpha : tensor<1x1024x15x15xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, f32)
        outs(%init : tensor<1x1024x7x7xf32>)
        -> tensor<1x1024x7x7xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK-SAME: ins(%[[bias]] :

    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :

    // CHECK: %[[lrelu:.+]] = linalg.lrelu_2d_nchw
    // CHECK-SAME: ins(%[[conv]], %[[alpha]] :
    // CHECK-SAME: outs(%[[conv]] :

    // CHECK: %[[padded:.+]] = tensor.pad %[[lrelu]] low[0, 0, 0, 0] high[0, 0, 1, 1]
    // CHECK: tensor.yield %[[pad_value]] : f32

    // CHECK: %[[pool:.+]] = tensor.empty()

    // CHECK: %[[out:.+]] = linalg.pooling_nchw_max
    // CHECK: ins(%[[padded]], %[[pool]] :

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x7x7xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_relu_maxpool(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
func.func @unfuse_conv_2d_relu_maxpool(%ifm : tensor<1x1024x15x15xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>

    // CHECK-DAG: %[[pad_value:.+]] = arith.constant 0xFF800000 : f32

    %init = tensor.splat %zero : tensor<1x1024x7x7xf32>
    %result = linalg.conv_2d_relu_maxpool
        {
            dilations = dense<1> : tensor<2xi64>,
            strides = dense<1> : tensor<2xi64>,
            mpKernelSize = dense<2> : tensor<2xi64>,
            mpStrides = dense<2> : tensor<2xi64>,
            mpDilations = dense<1> : tensor<2xi64>,
            mpPadding = dense<[0, 1, 0, 1]> : tensor<4xi64>
        }
        ins(%ifm, %weights, %bias : tensor<1x1024x15x15xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>)
        outs(%init : tensor<1x1024x7x7xf32>)
        -> tensor<1x1024x7x7xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK-SAME: ins(%[[bias]] :

    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :

    // CHECK: %[[lrelu:.+]] = linalg.relu_2d_nchw
    // CHECK-SAME: ins(%[[conv]] :
    // CHECK-SAME: outs(%[[conv]] :

    // CHECK: %[[padded:.+]] = tensor.pad %[[lrelu]] low[0, 0, 0, 0] high[0, 0, 1, 1]
    // CHECK: tensor.yield %[[pad_value]] : f32

    // CHECK: %[[pool:.+]] = tensor.empty()

    // CHECK: %[[out:.+]] = linalg.pooling_nchw_max
    // CHECK: ins(%[[padded]], %[[pool]] :

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x7x7xf32>
}

// -----

// CHECK-DAG: #[[sumIn:.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK-DAG: #[[sumOut:.+]] = affine_map<(d0, d1, d2) -> (d0, 0, d1)>
// CHECK-DAG: #[[divIn:.+]] = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
// CHECK-DAG: #[[divOut:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @unfuse_softmax(
// CHECK-SAME: %[[ifm:.+]]: tensor<4x3x2xf32>
func.func @unfuse_softmax(%ifm : tensor<4x3x2xf32>) -> tensor<4x3x2xf32> {
    // BUG: linalg.softmax does not have a proper assembly format with $dim!
    %result = "linalg.softmax" (%ifm) {dim = -2 : i64} : (tensor<4x3x2xf32>) -> (tensor<4x3x2xf32>)

    // CHECK: %[[exp:.+]] = math.exp %[[ifm]]

    // CHECK: %[[sum:.+]] = linalg.generic
    // CHECK: indexing_maps = [#[[sumIn]], #[[sumOut]]]
    // CHECK: iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK: ins(%[[exp]] :

    // CHECK: %[[out:.+]] = linalg.generic
    // CHECK: indexing_maps = [#[[divIn]], #[[divOut]]]
    // CHECK: iterator_types = ["parallel", "parallel", "parallel"]
    // CHECK: ins(%[[sum]]
    // CHECK: outs(%[[exp]]

    // CHECK: return %[[out]]
    return %result : tensor<4x3x2xf32>
}

// -----

// CHECK: func @unfuse_globalaveragepool2d(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x2048x7x7xf32>
func.func @unfuse_globalaveragepool2d(%ifm : tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32> {

    %result = linalg.globalaveragepool2d ins(%ifm : tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32>

    // CHECK: %[[accu:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2048x1x1xf32>
    // CHECK: %[[div:.+]] = arith.constant dense<4.900000e+01> : tensor<1x2048x1x1xf32>
    // CHECK: %[[krnl:.+]] = tensor.empty() : tensor<7x7xf32>
    // CHECK: %[[sum:.+]] = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %0 : tensor<1x2048x7x7xf32>, tensor<7x7xf32>) outs(%[[accu]] : tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
    // CHECK: %[[out:.+]] = arith.divf %[[sum]], %[[div]] : tensor<1x2048x1x1xf32>

    // CHECK: return %[[out]]

    return %result : tensor<1x2048x1x1xf32>
}

// -----

// CHECK: func @unfuse_linear(
// CHECK-SAME: %[[input:.+]]: tensor<1x2048xf32>, %[[weights:.+]]: tensor<1000x2048xf32>, %[[bias:.+]]: tensor<1000xf32>
func.func @unfuse_linear(%input: tensor<1x2048xf32>, %weights: tensor<1000x2048xf32>, %bias: tensor<1000xf32>) -> tensor<1x1000xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.splat %zero : tensor<1x1000xf32>
    %result = linalg.linear ins(%input, %weights, %bias: tensor<1x2048xf32>, tensor<1000x2048xf32>, tensor<1000xf32>) outs(%init: tensor<1x1000xf32>) -> tensor<1x1000xf32>

// CHECK:  %[[tweightshape:.+]] = tensor.empty() : tensor<2048x1000xf32>
// CHECK:  %[[tweights:.+]] = linalg.transpose2d ins(%arg1 : tensor<1000x2048xf32>) outs(%0 : tensor<2048x1000xf32>) -> tensor<2048x1000xf32>
// CHECK:  %[[bias2dshape:.+]] = tensor.empty() : tensor<1x1000xf32>
// CHECK:  %[[bias2d:.+]] = linalg.broadcast_1d_to_2d ins(%arg2 : tensor<1000xf32>) outs(%2 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
// CHECK:  %[[out:.+]] = linalg.matmul ins(%[[input]], %[[tweights]] : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%[[bias2d]] : tensor<1x1000xf32>) -> tensor<1x1000xf32
// CHECK: return %[[out]]

    return %result : tensor<1x1000xf32>
}

// -----

// CHECK:  func.func @unfuse_linearRelu
// CHECK-SAME: %[[input:.+]]: tensor<1x2048xf32>, %[[weights:.+]]: tensor<1000x2048xf32>, %[[bias:.+]]: tensor<1000xf32>
func.func @unfuse_linearRelu(%input: tensor<1x2048xf32>, %weights: tensor<1000x2048xf32>, %bias: tensor<1000xf32>) -> tensor<1x1000xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.splat %zero : tensor<1x1000xf32>
    %result = linalg.linear_relu ins(%input, %weights, %bias: tensor<1x2048xf32>, tensor<1000x2048xf32>, tensor<1000xf32>) outs(%init: tensor<1x1000xf32>) -> tensor<1x1000xf32>

// CHECK:  %[[tweightshape:.+]] = tensor.empty() : tensor<2048x1000xf32>
// CHECK:  %[[tweights:.+]] = linalg.transpose2d ins(%arg1 : tensor<1000x2048xf32>) outs(%0 : tensor<2048x1000xf32>) -> tensor<2048x1000xf32>
// CHECK:  %[[bias2dshape:.+]] = tensor.empty() : tensor<1x1000xf32>
// CHECK:  %[[bias2d:.+]] = linalg.broadcast_1d_to_2d ins(%arg2 : tensor<1000xf32>) outs(%2 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
// CHECK:  %[[matmul:.+]] = linalg.matmul ins(%[[input]], %[[tweights]] : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%[[bias2d]] : tensor<1x1000xf32>) -> tensor<1x1000xf32
// CHECK:  %[[out:.*]] = linalg.relu_nc ins(%[[matmul]] : tensor<1x1000xf32>) outs(%[[matmul]] : tensor<1x1000xf32>) -> tensor<1x1000xf32>
// CHECK: return %[[out]]

    return %result : tensor<1x1000xf32>
}