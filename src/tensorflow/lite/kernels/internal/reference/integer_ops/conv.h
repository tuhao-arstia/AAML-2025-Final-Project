/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "cfu.h"
#include "perf.h"

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
    
    // Do not modify the code below this line.
    perf_enable_counter(6);
    const int32_t input_offset = params.input_offset;
    const int32_t output_offset = params.output_offset;
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    // Do not modify the code above this line.
    
    // My im2col implementation:
    // Get parameters.
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
        
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int output_depth = output_shape.Dims(3);
        
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);

    // im2col parameters.
    const int M = output_height * output_width;
    const int K = filter_height * filter_width * input_depth;
    const int N = output_depth;

    // Given the parameters from the model
    int32_t Matrix_A [5000000]; // this size should be max(M*K/4) of the model. 
    int32_t Matrix_B [5000000]; // this size should be max(K*N/4) of the model. 

    // ---------------------------------------------------------------------------
    // Task 1 : Im2Col Packing
    //  image -> Matrix_A
    // filter -> Matrix_B
    // ---------------------------------------------------------------------------
    int idx_A = 0;
    int idx_B = 0;

    // --- Pack Matrix A (Input) ---
    for (int m_base = 0; m_base < M; m_base += 4) {
        // Optimization 1: Pre-calculate coordinates for the 4 pixels in this strip.
        // This removes (K * 4) divisions/modulos per strip.
        int oy[4], ox[4];
        bool valid_m[4];

        for (int i = 0; i < 4; ++i) {
            int m = m_base + i;
            if (m < M) {
                valid_m[i] = true;
                oy[i] = m / output_width;
                ox[i] = m % output_width;
            } else {
                valid_m[i] = false;
                oy[i] = 0; ox[i] = 0; // Dummy values
            }
        }

        // Optimization 2: Use counters for K decomposition
        int k_fy = 0;
        int k_fx = 0;
        int k_fd = 0;

        for (int k = 0; k < K; ++k) {
            int32_t packed_val = 0;
            
            // Unrolled packing loop (i = 0, 1, 2, 3)
            // Manual unrolling allows us to use the pre-calculated oy/ox arrays
            for (int i = 0; i < 4; ++i) {
                // Default padding value
                int8_t val = (int8_t)(-input_offset);
                
                if (valid_m[i]) {
                    const int in_y = oy[i] * stride_height + k_fy * dilation_height_factor - pad_height;
                    const int in_x = ox[i] * stride_width + k_fx * dilation_width_factor - pad_width;
                    
                    const bool is_point_inside_image =
                        (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);
                    
                    if (is_point_inside_image) {
                        val = input_data[Offset(input_shape, 0, in_y, in_x, k_fd)];
                    }
                }
                int shift = (3 - i) * 8; 
                packed_val |= ((static_cast<uint32_t>(val) & 0xFF) << shift);
            }
          
            Matrix_A[idx_A] = packed_val;
            idx_A++;
          
            // Update K counters (Strength Reduction: ++ instead of / %)
            k_fd++;
            if (k_fd == input_depth) {
                k_fd = 0;
                k_fx++;
                if (k_fx == filter_width) {
                    k_fx = 0;
                    k_fy++;
                }
            }
        }
    }

    // --- Pack Matrix B (Filter) ---
    for (int n_base = 0; n_base < N; n_base += 4) {
        // Reset K counters for Matrix B loop
        int k_fy = 0;
        int k_fx = 0;
        int k_fd = 0;

        for (int k = 0; k < K; ++k) {
            int32_t packed_val = 0;
            
            for (int i = 0; i < 4; ++i) {
                int out_channel = n_base + i;
                int8_t val = 0;
                
                if (out_channel < N) {
                    val = filter_data[Offset(filter_shape, out_channel, k_fy, k_fx, k_fd)];
                }
                packed_val |= ((static_cast<uint32_t>(val) & 0xFF) << (i * 8));
            }
          
            Matrix_B[idx_B] = packed_val;
            idx_B++;
          
            // Update K counters
            k_fd++;
            if (k_fd == input_depth) {
                k_fd = 0;
                k_fx++;
                if (k_fx == filter_width) {
                    k_fx = 0;
                    k_fy++;
                }
            }
        }
    }

    // Tiling parameters for CFU matmul (oringal KMN=150,128,100)
    const int TILE_K = 150;
    const int TILE_M = 128; // TILE_M must be multiple of 4
    const int TILE_N = 100; // TILE_N must be multiple of 4
    // ---------------------------------------------------------------------------
    // Task 2: Send KMN and input_offset Parameter
    // ---------------------------------------------------------------------------
    uint32_t kmn_packed = (TILE_K << 16) | (TILE_M << 8) | TILE_N;
    cfu_op0(2, kmn_packed, input_offset);

    // ---------------------------------------------------------------------------
    // Task 3: Tiled Convolution Execution
    // ---------------------------------------------------------------------------
    // Number of strips per tile (for loop bounds)
    // Matrix A Strip: 4 rows. Tile A has 128 rows -> 32 strips.
    const int STRIPS_A = TILE_M / 4;       
    // Matrix B Strip: 4 cols. Tile B has 100 cols -> 25 strips.
    const int STRIPS_B = TILE_N / 4; 

    // Prepare Padding Value for Matrix A (Input)
    // Hardware adds input_offset, so we must send -input_offset to result in 0 accumulation
    int32_t packed_padding_val_A = 0;
    for(int i=0; i<4; ++i) {
        packed_padding_val_A |= ((static_cast<uint32_t>((int8_t)(-input_offset)) & 0xFF) << (i*8));
    }

    for (int m_base = 0; m_base < M; m_base += TILE_M) {
        for (int n_base = 0; n_base < N; n_base += TILE_N) {

            // --- Computation Loop (K) ---
            for (int k_base = 0; k_base < K; k_base += TILE_K) {

                // Stream A
                for (int s = 0; s < STRIPS_A; ++s) {
                    int global_strip_idx = (m_base / 4) + s;
                    bool is_strip_valid = (global_strip_idx < (M + 3) / 4);
                    for (int k = 0; k < TILE_K; ++k) {
                        int32_t val = (is_strip_valid && (k_base + k < K)) ? 
                                    Matrix_A[global_strip_idx * K + (k_base + k)] : packed_padding_val_A;
                        cfu_op0(0, val, 0);
                    }
                }
              
                // Stream B
                for (int s = 0; s < STRIPS_B; ++s) {
                    int global_strip_idx = (n_base / 4) + s;
                    bool is_strip_valid = (global_strip_idx < (N + 3) / 4);
                    for (int k = 0; k < TILE_K; ++k) {
                        int32_t val = (is_strip_valid && (k_base + k < K)) ? 
                                    Matrix_B[global_strip_idx * K + (k_base + k)] : 0;
                        cfu_op0(1, val, 0);
                    }
                }
              
                // Compute
                cfu_op0(3, (k_base > 0) ? 1 : 0, 0);
            }
          
            // --- Readout Loop (Correction Applied) ---
            for (int s = 0; s < STRIPS_B; ++s) {
                for (int r = 0; r < TILE_M; ++r) {

                    int32_t acc_vals[4];
                    acc_vals[3] = cfu_op0(4, 0, 0); 
                    acc_vals[2] = cfu_op0(4, 0, 0);
                    acc_vals[1] = cfu_op0(4, 0, 0);
                    acc_vals[0] = cfu_op0(4, 0, 0);
                    
                    int global_m = m_base + r;
                    
                    for (int i = 0; i < 4; ++i) {
                        int n_local = s * 4 + i;
                        int global_n = n_base + n_local;
                        
                        if (global_m < M && global_n < N && n_local < TILE_N) {
                            int32_t acc = acc_vals[i];
                            if (bias_data) acc += bias_data[global_n];
                            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[global_n], output_shift[global_n]);
                            acc += output_offset;
                            acc = std::max(acc, output_activation_min);
                            acc = std::min(acc, output_activation_max);
                            
                            int out_y = global_m / output_width;
                            int out_x = global_m % output_width;
                            output_data[Offset(output_shape, 0, out_y, out_x, global_n)] = static_cast<int8_t>(acc);
                        }
                    }
                }
            }
        }
    }

    // ==============================================================
    //                 End of im2col implementation
    // ==============================================================
    perf_disable_counter(6);
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
