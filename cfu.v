// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);
  
  // Parameter
  parameter IDLE  = 2'd0;
  parameter INPUT = 2'd1;
  parameter CALC  = 2'd2;

  parameter ADDR_BITS_A = 13; // 8192
  parameter ADDR_BITS_B = 13;
  parameter ADDR_BITS_C = 13;

  // Declaration
  // bram buffer signals
  reg A_wr_en, B_wr_en, C_wr_en;
  reg [ADDR_BITS_A-1:0] A_index;
  reg [ADDR_BITS_B-1:0] B_index;
  reg [ADDR_BITS_C-1:0] C_index;
  reg [31:0] A_data_in, B_data_in;
  reg [127:0] C_data_in;
  wire [31:0] A_data_out, B_data_out;
  wire [127:0] C_data_out;

  // KMN signals
  reg [7:0] k, m, n;

  // TPU signals
  reg [1:0] cs, ns;
  wire [7:0] total_k, total_m, total_n;
  reg [7:0] cnt_k, ns_cnt_k;
  reg [7:0] cnt_m, ns_cnt_m;
  reg [7:0] cnt_n, ns_cnt_n;
  reg [1:0] cnt_wr, ns_cnt_wr;
  reg [1:0] cnt_rd, ns_cnt_rd;

  // SA signals
  wire sa_in_valid;
  reg signed [8:0] a_offset; // from cfu_op0(2, KMN, input offset)
  reg signed [9:0] a_00_in, a_10_in, a_20_in, a_30_in;
  reg signed [7:0] b_00_in, b_01_in, b_02_in, b_03_in;
  wire signed [31:0] acc_00, acc_01, acc_02, acc_03;
  wire signed [31:0] acc_10, acc_11, acc_12, acc_13;
  wire signed [31:0] acc_20, acc_21, acc_22, acc_23;
  wire signed [31:0] acc_30, acc_31, acc_32, acc_33;

  // C row buffer for accumulation
  reg c_buffer_en;
  reg signed [127:0] c_buffer [0:3];


  // bram buffer instantiation
  global_buffer_bram #(
    .ADDR_BITS(ADDR_BITS_A),  // ADDR_BITS 13 -> generates 2^13 = 8192 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_A(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(A_wr_en),
    .index(A_index),
    .data_in(A_data_in),
    .data_out(A_data_out)
  );

  global_buffer_bram #(
    .ADDR_BITS(ADDR_BITS_B),  // ADDR_BITS 13 -> generates 2^13 = 8192 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_B(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(B_wr_en),
    .index(B_index),
    .data_in(B_data_in),
    .data_out(B_data_out)
  );

  global_buffer_bram #(
    .ADDR_BITS(ADDR_BITS_C),  // ADDR_BITS 13 -> generates 2^13 = 8192 entries
    .DATA_BITS(128)  // DATA_BITS 128 -> 128 bits for each entries
  )
  gbuff_C(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(C_wr_en),
    .index(C_index),
    .data_in(C_data_in),
    .data_out(C_data_out)
  );

  // bram buffer A and B logic (read)
  always @(*) begin
    // default: READ
    A_wr_en = 1'b0;
    B_wr_en = 1'b0;
    
    if(cmd_valid)begin
        case (cmd_payload_function_id[9:3])
            // write data to buffer A
            7'd0: begin
                A_wr_en = 1'b1;
            end
            // write data to buffer B
            7'd1: begin
                B_wr_en = 1'b1;
            end
            default: begin
                A_wr_en = 1'b0;
                B_wr_en = 1'b0;
            end
        endcase
    end
  end
  // Buffer A index logic
  always @(posedge clk) begin
    if(reset)begin
        A_index <= 0;
    end else if(cmd_valid) begin
        case (cmd_payload_function_id[9:3])
            // read data and write to buffer A
            7'd0: begin
                A_index <= A_index + 'd1;
            end
            default: begin
                A_index <= 0;
            end
        endcase
    end else if (cs == INPUT)begin
        A_index <= 0;
    end else if (cs == CALC) begin
        if(cnt_k == total_k) begin
            if(cnt_m == total_m) begin
                A_index <= 0;
            end 
        end else if (cnt_k < k) begin
            A_index <= A_index + 'd1;
        end
    end
  end
  // Buffer B index logic
  always @(posedge clk) begin
    if(reset)begin
        B_index <= 0;
    end else if(cmd_valid) begin
        case (cmd_payload_function_id[9:3])
            // read data and write to buffer B
            7'd1: begin
                B_index <= B_index + 'd1;
            end
            default: begin
                B_index <= 0;
            end
        endcase
    end else if (cs == INPUT)begin
        B_index <= 0;
    end else if (cs == CALC) begin
        if(cnt_k == total_k) begin
            if(cnt_m == total_m) begin
                if(cnt_n == total_n) begin
                    B_index <= 0;
                end else begin
                    B_index <= k * (cnt_n + 'd1);
                end
            end else begin
                B_index <= k * cnt_n;
            end
        end else if(cnt_k < k) begin
            B_index <= B_index + 'd1;
        end
    end
  end

  always @(*) begin
    A_data_in = cmd_payload_inputs_0;
    B_data_in = cmd_payload_inputs_0;
  end

  // KMN logic
  always @(posedge clk) begin
    if(reset) begin
        k <= 8'd0;
        m <= 8'd0;
        n <= 8'd0;
    end else if(cmd_valid) begin
        case (cmd_payload_function_id[9:3])
            7'd2: begin
                k <= cmd_payload_inputs_0[23:16];
                m <= cmd_payload_inputs_0[15:8];
                n <= cmd_payload_inputs_0[7:0];
            end
            default: begin
                k <= k;
                m <= m;
                n <= n;
            end
        endcase
    end
  end

  // total counter logic
  assign total_k = (reset) ? 0 : k + 8'd6;
  assign total_m = (reset) ? 0 : (m-1) >> 2;
  assign total_n = (reset) ? 0 : (n-1) >> 2;

  // MATMUL related logic
  // FSM-MATMUL
  // TPU FSM logic
  always @(posedge clk) begin
    if (reset) begin
        cs <= IDLE;
    end else begin
        cs <= ns;
    end
  end

  always @(*) begin
      ns = cs;
      case (cs)
          IDLE: begin
              // waiting for cfu_op == 3
              if (cmd_valid && cmd_payload_function_id[9:3] == 7'd3) begin
                  ns = INPUT;
              end
          end
          INPUT: begin
              ns = CALC;
          end
          CALC: begin
              if(cnt_k == total_k && cnt_m == total_m && cnt_n == total_n) begin
                  ns = IDLE;
              end
          end
          default: begin
              ns = cs;
          end
      endcase
  end

// k counter logic
always @(posedge clk) begin
    if(reset) begin
        cnt_k <= 8'd0;
    end else begin
        cnt_k <= ns_cnt_k;
  end
end
always @(*) begin
    ns_cnt_k = 0;
    case (cs)
        CALC: begin
            if (cnt_k == k + 8'd6) begin
                ns_cnt_k = 8'd0;
            end else begin
                ns_cnt_k = cnt_k + 8'd1;
            end
        end
    endcase
end

// m counter logic
always @(posedge clk) begin
    if(reset) begin
        cnt_m <= 8'd0;
    end else begin
        cnt_m <= ns_cnt_m;
    end
end
always @(*) begin
    ns_cnt_m = 0;
    case (cs)
        CALC: begin
            if (cnt_k == total_k) begin
                if(cnt_m == total_m) begin
                    ns_cnt_m = 8'd0;
                end else begin
                    ns_cnt_m = cnt_m + 8'd1;
                end
            end else begin
                ns_cnt_m = cnt_m;
            end
        end
    endcase
end

// n counter logic
always @(posedge clk) begin
    if(reset) begin
        cnt_n <= 8'd0;
    end else begin
        cnt_n <= ns_cnt_n;
    end
end
always @(*) begin
    ns_cnt_n = 0;
    case (cs)
        CALC: begin
            if (cnt_k == total_k) begin
                if(cnt_m == total_m) begin
                    if(cnt_n == total_n) begin
                        ns_cnt_n = 8'd0;
                    end else begin
                        ns_cnt_n = cnt_n + 8'd1;
                    end
                end else begin
                    ns_cnt_n = cnt_n;
                end
            end else begin
                ns_cnt_n = cnt_n;
            end
        end
    endcase
end

// SA instantiation
// The modified part !!!
assign sa_in_valid = (cs == INPUT);

SA u_SA (
    .clk(clk),
    .reset(reset),
    .in_valid(sa_in_valid),
    .k(k), .m(m), .n(n),
    .a_00(a_00_in), .a_10(a_10_in), .a_20(a_20_in), .a_30(a_30_in),
    .b_00(b_00_in), .b_01(b_01_in), .b_02(b_02_in), .b_03(b_03_in),
    .acc_00(acc_00), .acc_01(acc_01), .acc_02(acc_02), .acc_03(acc_03),
    .acc_10(acc_10), .acc_11(acc_11), .acc_12(acc_12), .acc_13(acc_13),
    .acc_20(acc_20), .acc_21(acc_21), .acc_22(acc_22), .acc_23(acc_23),
    .acc_30(acc_30), .acc_31(acc_31), .acc_32(acc_32), .acc_33(acc_33)
);

// cfu_op0(2) input_offset logic
always @(posedge clk) begin
    if (reset) begin
        a_offset <= 0;
    end else if (cmd_valid) begin
        case (cmd_payload_function_id[9:3])
            7'd2: begin
                a_offset <= cmd_payload_inputs_1[8:0];
            end
        endcase
    end
end

always @(*) begin
    a_00_in = 0; a_10_in = 0; a_20_in = 0; a_30_in = 0;
    b_00_in = 0; b_01_in = 0; b_02_in = 0; b_03_in = 0;
    case (cs)
        CALC: begin
            if (cnt_k < k) begin
                a_00_in = $signed(A_data_out[31:24]) + $signed(a_offset); 
                a_10_in = $signed(A_data_out[23:16]) + $signed(a_offset); 
                a_20_in = $signed(A_data_out[15:8]) + $signed(a_offset); 
                a_30_in = $signed(A_data_out[7:0]) + $signed(a_offset);

                b_00_in = B_data_out[31:24]; 
                b_01_in = B_data_out[23:16]; 
                b_02_in = B_data_out[15:8]; 
                b_03_in = B_data_out[7:0];
            end
        end
    endcase
end

// C buffer logic
// write counter for C buffer
always @(posedge clk) begin
    if (reset) begin
        cnt_wr <= 2'd0;
    end else begin
        cnt_wr <= ns_cnt_wr;
    end
end
always @(*) begin
    ns_cnt_wr = 0;
    case (cs)
        CALC: begin
            if(cnt_k >= k + 8'd3) begin
                ns_cnt_wr = cnt_wr + 2'd1;
            end
        end
    endcase
end

// write enable for C buffer
always @(*) begin
    C_wr_en = 0;
    case (cs)
        CALC: begin
            if(cnt_k >= k + 8'd3) begin
                if(cnt_m == total_m) begin
                    case (m[1:0])
                        2'b00: begin
                            C_wr_en = 1;
                        end
                        2'b01: begin
                            C_wr_en = (cnt_wr == 2'b00) ? 1 : 0;
                        end
                        2'b10: begin
                            C_wr_en = (cnt_wr < 2'b10) ? 1 : 0;
                        end
                        2'b11: begin
                            C_wr_en = (cnt_wr < 2'b11) ? 1 : 0;
                        end
                    endcase
                end else begin
                    C_wr_en = 1;
                end
            end
        end
    endcase
end

// C index logic
always @(posedge clk) begin
    if (reset) begin
        C_index <= 0;
    end else if (cmd_valid) begin
      case (cmd_payload_function_id[9:3])
          7'd4:begin
              if(cnt_rd == 2'd3) begin
                  C_index <= C_index + 'd1;
              end else begin
                  C_index <= C_index;
              end
          end
          default: begin
              C_index <= 0;
          end
      endcase
    end else begin
        case (cs)
            INPUT: begin
                C_index <= 0;
            end
            CALC: begin
                if( cnt_k >= k-1 && cnt_k < k+2 )begin
                    C_index <= C_index + 1;
                end else if (cnt_k == k+2 ) begin
                    C_index <= C_index - 3;
                end else if (cnt_k >= k+3)begin
                    if(cnt_k == total_k && cnt_m == total_m && cnt_n == total_n) begin
                        C_index <= 0;
                    end else begin
                        C_index <= C_index + C_wr_en;
                    end
                end
            end
        endcase
    end
end

// c row buffer enable : for k tiling accumulation
always @(posedge clk) begin
    if (reset) begin
        c_buffer_en <= 1'b0;
    end else if (cmd_valid) begin
        case (cmd_payload_function_id[9:3])
            7'd3: begin
                c_buffer_en <= cmd_payload_inputs_0[0];
            end  
        endcase
    end
end
// c row buffers
always @(posedge clk) begin
    if(reset) begin
        c_buffer[0] <= 0;
    end else if (cs == CALC) begin
        if(c_buffer_en)begin
            if(cnt_k == k-1)begin
                c_buffer[0] <= C_data_out;
            end
        end else begin
            c_buffer[0] <= 0;
        end
    end
end

always @(posedge clk) begin
    if(reset) begin
        c_buffer[1] <= 0;
    end else if (cs == CALC) begin
        if(c_buffer_en)begin
            if(cnt_k == k)begin
                c_buffer[1] <= C_data_out;
            end
        end else begin
            c_buffer[1] <= 0;
        end
    end
end

always @(posedge clk) begin
    if(reset) begin
        c_buffer[2] <= 0;
    end else if (cs == CALC) begin
        if(c_buffer_en)begin
            if(cnt_k == k+1)begin
                c_buffer[2] <= C_data_out;
            end
        end else begin
            c_buffer[2] <= 0;
        end
    end
end

always @(posedge clk) begin
    if(reset) begin
        c_buffer[3] <= 0;
    end else if (cs == CALC) begin
        if(c_buffer_en)begin
            if(cnt_k == k+2)begin
                c_buffer[3] <= C_data_out;
            end
        end else begin
            c_buffer[3] <= 0;
        end
    end
end

// C data in logic
always @(*) begin
    C_data_in = 0;
    case (cs)
        CALC: begin
            if(cnt_k >= k + 8'd3) begin
                case (cnt_wr)
                    2'b00: begin
                        C_data_in[127:96] = $signed(acc_00) + $signed(c_buffer[0][127:96]);
                        C_data_in [95:64] = $signed(acc_01) + $signed(c_buffer[0] [95:64]);
                        C_data_in [63:32] = $signed(acc_02) + $signed(c_buffer[0] [63:32]);
                        C_data_in  [31:0] = $signed(acc_03) + $signed(c_buffer[0]  [31:0]);
                    end
                    2'b01: begin
                        C_data_in[127:96] = $signed(acc_10) + $signed(c_buffer[1][127:96]);
                        C_data_in [95:64] = $signed(acc_11) + $signed(c_buffer[1] [95:64]);
                        C_data_in [63:32] = $signed(acc_12) + $signed(c_buffer[1] [63:32]);
                        C_data_in  [31:0] = $signed(acc_13) + $signed(c_buffer[1]  [31:0]);
                    end
                    2'b10: begin
                        C_data_in[127:96] = $signed(acc_20) + $signed(c_buffer[2][127:96]);
                        C_data_in [95:64] = $signed(acc_21) + $signed(c_buffer[2] [95:64]);
                        C_data_in [63:32] = $signed(acc_22) + $signed(c_buffer[2] [63:32]);
                        C_data_in  [31:0] = $signed(acc_23) + $signed(c_buffer[2]  [31:0]);
                    end
                    2'b11: begin
                        C_data_in[127:96] = $signed(acc_30) + $signed(c_buffer[3][127:96]);
                        C_data_in [95:64] = $signed(acc_31) + $signed(c_buffer[3] [95:64]);
                        C_data_in [63:32] = $signed(acc_32) + $signed(c_buffer[3] [63:32]);
                        C_data_in  [31:0] = $signed(acc_33) + $signed(c_buffer[3]  [31:0]);
                    end
                endcase
            end
        end
    endcase
end

// counter for reading C data
always @(posedge clk) begin
    if (reset) begin
        cnt_rd <= 2'd0;
    end else begin
        cnt_rd <= ns_cnt_rd;
    end
end
always @(*) begin
    ns_cnt_rd = cnt_rd;
    if (cmd_valid) begin
        case (cmd_payload_function_id[9:3])
            7'd4: begin
                ns_cnt_rd = cnt_rd + 2'd1;
            end
            default: begin
                ns_cnt_rd = 0;
            end
        endcase
    end
end

  // handshake signal
assign cmd_ready = ~rsp_valid;

always @(posedge clk) begin
    if (reset) begin
        rsp_valid <= 1'b0;
    end else if (rsp_valid) begin
        rsp_valid <= ~rsp_ready;
    end else if (cmd_valid) begin
        if(cmd_payload_function_id[9:3] == 7'd3) begin
            rsp_valid <= 1'b0;
        end else begin
            rsp_valid <= 1'b1;
        end
    end else if (cs == CALC && cnt_k == total_k && cnt_m == total_m && cnt_n == total_n) begin
        rsp_valid <= 1'b1;
    end
end

  always @(posedge clk) begin
    if (reset) begin
        rsp_payload_outputs_0 <= 0;
    end else if (cmd_valid) begin
        case (cmd_payload_function_id[9:3])
            7'd4: begin
                case (cnt_rd)
                    2'd0: begin
                        rsp_payload_outputs_0 <= C_data_out[127:96];
                    end
                    2'd1: begin
                        rsp_payload_outputs_0 <= C_data_out[95:64];
                    end
                    2'd2: begin
                        rsp_payload_outputs_0 <= C_data_out[63:32];
                    end
                    2'd3: begin
                        rsp_payload_outputs_0 <= C_data_out[31:0];
                    end 
                endcase
            end
            default: begin
                // case:
                // 0. read data and write to bram buffer A (no output)
                // 1. read data and write to bram buffer B (no output)
                // 2. pass KMN parameters (no output)
                // 3. calculate matmul and store result in bram buffer C (no output)
                rsp_payload_outputs_0 <= 0;
            end
        endcase
    end
  end

endmodule

// =============================================== //
//                   Submodules                    //
// =============================================== //
// Global Buffer BRAM Module
module global_buffer_bram #(parameter ADDR_BITS=8, parameter DATA_BITS=8)(
  input                      clk,
  input                      rst_n,
  input                      ram_en,
  input                      wr_en,
  input      [ADDR_BITS-1:0] index,
  input      [DATA_BITS-1:0] data_in,
  output reg [DATA_BITS-1:0] data_out
  );

  parameter DEPTH = 2**ADDR_BITS;

  reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

  always @ (negedge clk) begin
    if (ram_en) begin
      if(wr_en) begin
        gbuff[index] <= data_in;
      end
      else begin
        data_out <= gbuff[index];
      end
    end
  end

endmodule

// Processing Element (PE) Module
module PE (
    clk,
    reset,
    in_valid,
    k,
    a_in,
    b_in,
    a_out,
    b_out,
    acc_out
);
input clk;
input reset;
input in_valid;
input [7:0] k;
input signed [9:0] a_in;
input signed [7:0] b_in;
output reg signed [9:0] a_out;
output reg signed [7:0] b_out;
output reg signed [31:0] acc_out;

// counter for acc reset
reg [7:0] cnt_pe;

// mult
wire signed [17:0] mult_ab;
assign mult_ab = $signed(a_in) * $signed(b_in);

// counter
always @(posedge clk) begin
    if(reset) begin
        cnt_pe <= 8'd0;
    end else if (in_valid) begin
        cnt_pe <= 8'd0;
    end else if (cnt_pe == k + 8'd6) begin
        cnt_pe <= 8'd0;
    end else begin
        cnt_pe <= cnt_pe + 'd1;
    end
end

always @(posedge clk) begin
    if(reset) begin
        a_out <= 0;
        b_out <= 0;
    end else begin
        a_out <= a_in;
        b_out <= b_in;
    end
end

always @(posedge clk) begin
    if(reset) begin
        acc_out <= 32'd0;
    end else if (cnt_pe == k + 8'd6) begin
        acc_out <= 32'd0;
    end else begin
        acc_out <= $signed(acc_out) + $signed(mult_ab);
    end
end 

endmodule

// Systolic Array (4x4) Module
module SA (
    clk,
    reset,
    in_valid,
    // input specifications
    k, m, n,
    // input elements
    a_00, a_10, a_20, a_30,
    b_00, b_01, b_02, b_03,
    // output accumulations
    acc_00, acc_01, acc_02, acc_03,
    acc_10, acc_11, acc_12, acc_13,
    acc_20, acc_21, acc_22, acc_23,
    acc_30, acc_31, acc_32, acc_33
);

input clk;
input reset;
input in_valid;
input [7:0] k, m, n;
input signed [9:0] a_00, a_10, a_20, a_30;
input signed [7:0] b_00, b_01, b_02, b_03;
output signed [31:0] acc_00, acc_01, acc_02, acc_03;
output signed [31:0] acc_10, acc_11, acc_12, acc_13;
output signed [31:0] acc_20, acc_21, acc_22, acc_23;
output signed [31:0] acc_30, acc_31, acc_32, acc_33;

// SA internal wires and regs
wire signed [9:0] a_01, a_02, a_03, a_04;
wire signed [9:0] a_11, a_12, a_13, a_14;
wire signed [9:0] a_21, a_22, a_23, a_24;
wire signed [9:0] a_31, a_32, a_33, a_34;
wire signed [7:0] b_10, b_11, b_12, b_13;
wire signed [7:0] b_20, b_21, b_22, b_23;
wire signed [7:0] b_30, b_31, b_32, b_33;
wire signed [7:0] b_40, b_41, b_42, b_43;

reg signed [9:0] a_10_p1;
reg signed [9:0] a_20_p1, a_20_p2;
reg signed [9:0] a_30_p1, a_30_p2, a_30_p3;
reg signed [7:0] b_01_p1;
reg signed [7:0] b_02_p1, b_02_p2;
reg signed [7:0] b_03_p1, b_03_p2, b_03_p3;

// implementation
// input pipeline
always @(posedge clk) begin
    if(reset) begin
        a_10_p1 <= 0;
        a_20_p1 <= 0;
        a_20_p2 <= 0;
        a_30_p1 <= 0;
        a_30_p2 <= 0;
        a_30_p3 <= 0;
        b_01_p1 <= 0;
        b_02_p1 <= 0;
        b_02_p2 <= 0;
        b_03_p1 <= 0;
        b_03_p2 <= 0;
        b_03_p3 <= 0;
    end else begin
        a_10_p1 <= a_10;
        a_20_p1 <= a_20;
        a_20_p2 <= a_20_p1;
        a_30_p1 <= a_30;
        a_30_p2 <= a_30_p1;
        a_30_p3 <= a_30_p2;
        b_01_p1 <= b_01;
        b_02_p1 <= b_02;
        b_02_p2 <= b_02_p1;
        b_03_p1 <= b_03;
        b_03_p2 <= b_03_p1;
        b_03_p3 <= b_03_p2;
    end
end

// PE instantiation
PE PE_00 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_00), .b_in(b_00), .a_out(a_01), .b_out(b_10), .acc_out(acc_00) );
PE PE_01 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_01), .b_in(b_01_p1), .a_out(a_02), .b_out(b_11), .acc_out(acc_01) );
PE PE_02 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_02), .b_in(b_02_p2), .a_out(a_03), .b_out(b_12), .acc_out(acc_02) );
PE PE_03 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_03), .b_in(b_03_p3), .a_out(a_04), .b_out(b_13), .acc_out(acc_03) );
PE PE_10 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_10_p1), .b_in(b_10), .a_out(a_11), .b_out(b_20), .acc_out(acc_10) );
PE PE_11 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_11), .b_in(b_11), .a_out(a_12), .b_out(b_21), .acc_out(acc_11) );
PE PE_12 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_12), .b_in(b_12), .a_out(a_13), .b_out(b_22), .acc_out(acc_12) );
PE PE_13 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_13), .b_in(b_13), .a_out(a_14), .b_out(b_23), .acc_out(acc_13) );
PE PE_20 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_20_p2), .b_in(b_20), .a_out(a_21), .b_out(b_30), .acc_out(acc_20) );
PE PE_21 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_21), .b_in(b_21), .a_out(a_22), .b_out(b_31), .acc_out(acc_21) );
PE PE_22 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_22), .b_in(b_22), .a_out(a_23), .b_out(b_32), .acc_out(acc_22) );
PE PE_23 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_23), .b_in(b_23), .a_out(a_24), .b_out(b_33), .acc_out(acc_23) );
PE PE_30 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_30_p3), .b_in(b_30), .a_out(a_31), .b_out(b_40), .acc_out(acc_30) );
PE PE_31 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_31), .b_in(b_31), .a_out(a_32), .b_out(b_41), .acc_out(acc_31) );
PE PE_32 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_32), .b_in(b_32), .a_out(a_33), .b_out(b_42), .acc_out(acc_32) );
PE PE_33 ( .clk(clk), .reset(reset), .in_valid(in_valid), .k(k), .a_in(a_33), .b_in(b_33), .a_out(a_34), .b_out(b_43), .acc_out(acc_33) );

endmodule