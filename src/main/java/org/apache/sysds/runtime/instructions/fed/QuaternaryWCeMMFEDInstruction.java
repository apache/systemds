
package org.apache.sysds.runtime.instructions.fed;

import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;
import java.util.concurrent.Future;

public class QuaternaryWCeMMFEDInstruction extends QuaternaryFEDInstruction
{
  // input1 ... federated X
  // input2 ... U
  // input3 ... V
  // _input4 ... W
  protected QuaternaryWCeMMFEDInstruction(Operator operator,
    CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, String opcode, String instruction_str)
  {
    super(FEDType.Quaternary, operator, in1, in2, in3, in4, out, opcode, instruction_str);
  }

  @Override
  public void processInstruction(ExecutionContext execution_context)
  {
    MatrixObject matrix_object_1 = execution_context.getMatrixObject(input1);
    MatrixObject matrix_object_2 = execution_context.getMatrixObject(input2);
    MatrixObject matrix_object_3 = execution_context.getMatrixObject(input3);
    // TODO: include _input4

    if(matrix_object_1.isFederated() && !matrix_object_2.isFederated() && !matrix_object_3.isFederated())
    {
      FederatedRequest fed_req_1 = matrix_object_1.getFedMapping().broadcast(matrix_object_2);
      FederatedRequest fed_req_2 = matrix_object_1.getFedMapping().broadcast(matrix_object_3);
      FederatedRequest fed_req_3 = FederationUtils.callInstruction(instString, output,
        new CPOperand[]{input1, input2, input3},
        new long[]{matrix_object_1.getFedMapping().getID(), fed_req_1.getID(), fed_req_2.getID()});
      FederatedRequest fed_req_4 = matrix_object_1.getFedMapping().cleanup(getTID(), fed_req_1.getID());
      FederatedRequest fed_req_5 = matrix_object_1.getFedMapping().cleanup(getTID(), fed_req_2.getID());

      // execute federated instructions
      Future<FederatedResponse>[] tmp = matrix_object_1.getFedMapping().execute(getTID(), true, fed_req_1, fed_req_2, fed_req_3, fed_req_4, fed_req_5);

      // TODO: do something with the output

    }

    assert false: "Not implemented yet!\n";
  }

}