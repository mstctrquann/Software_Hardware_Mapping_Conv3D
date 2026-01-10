module processing_element #(
    parameter DATA_WIDTH = 32,      // 32-bit signed integers
    parameter ACC_WIDTH  = 32       // Accumulator width
) (
    // Clock and Reset
    input  wire                     clk,
    input  wire                     rst_n,          // Active-low reset
    // Control Signals
    input  wire                     enable,         // Enable MAC operation
    input  wire                     acc_clear,      // Clear accumulator
    // Data Inputs
    input  wire signed [DATA_WIDTH-1:0]  weight_in,      // Weight from SRAM
    input  wire signed [DATA_WIDTH-1:0]  activation_in,  // Input activation
    
    // Data Output
    output wire signed [ACC_WIDTH-1:0]   partial_sum_out // Accumulated result
);
    reg signed [2*DATA_WIDTH-1:0] product; //MAC unit output
    // Local accumulator register
    reg signed [ACC_WIDTH-1:0] accumulator;

    always @(weight_in or activation_in) begin
        if (enable) begin
            product = weight_in * activation_in;
        end else begin
            product = {(2*DATA_WIDTH){1'b0}}; 
        end
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= {ACC_WIDTH{1'b0}};
        end 
        else if (acc_clear) begin
            accumulator <= {ACC_WIDTH{1'b0}};
        end 
        else if (enable) begin
            accumulator <= accumulator + product[ACC_WIDTH-1:0];
        end
        // else: Hold current value
    end
    
    // Output Assignment
    assign partial_sum_out = accumulator;

endmodule