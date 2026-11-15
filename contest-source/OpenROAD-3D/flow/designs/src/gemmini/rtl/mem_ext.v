(* blackbox *)
module mem_ext (
  input wire [11:0] RW0_addr,
  input wire RW0_clk,
  input wire [127:0] RW0_wdata,
  output wire [127:0] RW0_rdata,
  input wire RW0_en,
  input wire RW0_wmode,
  input wire [15:0] RW0_wmask
);
endmodule
