(* blackbox *)
module mem_0_ext (
  input wire [8:0] R0_addr,
  input wire R0_clk,
  output wire [511:0] R0_data,
  input wire R0_en,
  input wire [8:0] W0_addr,
  input wire W0_clk,
  input wire [511:0] W0_data,
  input wire W0_en,
  input wire [63:0] W0_mask
);
endmodule
