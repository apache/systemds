package org.apache.sysml.runtime.instructions.gpu.context;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.IndexFunction;
import org.apache.sysml.runtime.functionobjects.ReduceAll;
import org.apache.sysml.runtime.functionobjects.ReduceCol;
import org.apache.sysml.runtime.functionobjects.ReduceRow;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.operators.*;
import org.apache.sysml.utils.Statistics;

/**
 * Implements aggregate unary instructions for CUDA
 */
public class AggregateUnaryGPUInstruction extends GPUInstruction {
  private CPOperand _input1 = null;
  private CPOperand _output = null;

  public AggregateUnaryGPUInstruction(Operator op, CPOperand in1, CPOperand out,
                                       String opcode, String istr)
  {
    super(op, opcode, istr);
    _gputype = GPUINSTRUCTION_TYPE.AggregateUnary;
    _input1 = in1;
    _output = out;
  }

  public static AggregateUnaryGPUInstruction parseInstruction(String str )
          throws DMLRuntimeException
  {
    String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
    String opcode = parts[0];
    CPOperand in1 = new CPOperand(parts[1]);
    CPOperand out = new CPOperand(parts[2]);

    // This follows logic similar to AggregateUnaryCPInstruction.
    // nrow, ncol & length should either read or refresh metadata
    Operator aggop = null;
    if(opcode.equalsIgnoreCase("nrow") || opcode.equalsIgnoreCase("ncol") || opcode.equalsIgnoreCase("length")) {
      throw new DMLRuntimeException("nrow, ncol & length should not be compiled as GPU instructions!");
    } else {
      aggop = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
    }
    return new AggregateUnaryGPUInstruction(aggop, in1, out, opcode, str);
  }

  @Override
  public void processInstruction(ExecutionContext ec)
          throws DMLRuntimeException
  {
    Statistics.incrementNoOfExecutedGPUInst();

    String opcode = getOpcode();

    // nrow, ncol & length should either read or refresh metadata
    if(opcode.equalsIgnoreCase("nrow") || opcode.equalsIgnoreCase("ncol") || opcode.equalsIgnoreCase("length")) {
      throw new DMLRuntimeException("nrow, ncol & length should not be compiled as GPU instructions!");
    }

    //get inputs
    MatrixObject in1 = ec.getMatrixInputForGPUInstruction(_input1.getName());

    int rlen = (int)in1.getNumRows();
    int clen = (int)in1.getNumColumns();

    LibMatrixCUDA.unaryAggregate(ec, in1, _output.getName(), (AggregateUnaryOperator)_optr);

    //release inputs/outputs
    ec.releaseMatrixInputForGPUInstruction(_input1.getName());

    // If the unary aggregate is a row reduction or a column reduction, it results in a vector
    // which needs to be released. Otherwise a scala is produced and it is copied back to the host
    // and set in the execution context by invoking the setScalarOutput
    IndexFunction indexFunction = ((AggregateUnaryOperator) _optr).indexFn;
    if (indexFunction instanceof ReduceRow || indexFunction instanceof ReduceCol) {
      ec.releaseMatrixOutputForGPUInstruction(_output.getName());
    }
  }

}
