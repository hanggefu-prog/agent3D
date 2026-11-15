// Logical compatibility layer for the shared RTL macro wrappers.  Every
// native cell below is a real ASAP7 FakeRAM LEF/Liberty master.  The legacy
// module name exists only at RTL elaboration and is flattened by synthesis.
// FakeRAM7 is whole-word write only, so a nonempty bit/byte mask enables the
// word; this is the same explicit approximation used by the qualified BP_FE
// ASAP7 frontend.
`define OPEN3D_FAKERAM_ADAPTER(LEGACY, NATIVE, ADDR_WIDTH, DATA_WIDTH) \
module LEGACY ( \
  input clk, input ce_in, input we_in, \
  input [ADDR_WIDTH-1:0] addr_in, \
  input [DATA_WIDTH-1:0] wd_in, input [DATA_WIDTH-1:0] w_mask_in, \
  output [DATA_WIDTH-1:0] rd_out \
); \
  NATIVE native_mem ( \
    .clk(clk), .ce_in(ce_in), .we_in(we_in & |w_mask_in), \
    .addr_in(addr_in), .wd_in(wd_in), .rd_out(rd_out) \
  ); \
endmodule

`OPEN3D_FAKERAM_ADAPTER(fakeram45_256x16,  fakeram7_256x16,  8,  16)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_512x64,  fakeram7_512x64,  9,  64)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_256x95,  fakeram7_256x95,  8,  95)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_64x7,    fakeram7_64x7,    6,   7)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_64x15,   fakeram7_64x15,   6,  15)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_64x96,   fakeram7_64x96,   6,  96)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_256x96,  fakeram7_256x96,  8,  96)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_32x64,   fakeram7_32x64,   5,  64)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_256x48,  fakeram7_256x48,  8,  48)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_32x32,   fakeram7_32x32,   5,  32)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_64x124,  fakeram7_64x124,  6, 124)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_64x62,   fakeram7_64x62,   6,  62)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_128x116, fakeram7_128x116, 7, 116)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_2048x39, fakeram7_2048x39, 11, 39)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_256x34,  fakeram7_256x34,  8,  34)
`OPEN3D_FAKERAM_ADAPTER(fakeram45_64x21,   fakeram7_64x21,   6,  21)

`undef OPEN3D_FAKERAM_ADAPTER
