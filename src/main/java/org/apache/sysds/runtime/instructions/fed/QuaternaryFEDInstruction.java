

package org.apache.sysds.runtime.instructions.fed;

import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class QuaternaryFEDInstruction extends ComputationFEDInstruction
{
  private CPOperand _input4 = null;
  private boolean _cacheU = false;
  private boolean _cacheV = false;

  public QuaternaryFEDInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4,
    CPOperand out, boolean cacheU, boolean cacheV, String opcode, String str)
  {
    super(FEDType.Quaternary, op, in1, in2, in3, out, opcode, str);
    _input4 = in4;
    _cacheU = cacheU;
    _cacheV = cacheV;
  }

  @Override
  public void processInstruction(ExecutionContext ec)
  {
    assert false: "Not implemented yet!";
  }

}
