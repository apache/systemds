/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.fed;

import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.QuaternaryOperator;

import java.util.concurrent.Future;

public class QuaternaryWSLossFEDInstruction extends QuaternaryFEDInstruction
{
  // input1 ... federated X
  // input2 ... U
  // input3 ... V
  // _input4 ... W
  protected QuaternaryWSLossFEDInstruction(Operator operator,
    CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, String opcode, String instruction_str)
  {
    super(FEDType.Quaternary, operator, in1, in2, in3, in4, out, opcode, instruction_str);
  }

  @Override
  public void processInstruction(ExecutionContext execution_context)
  {
    QuaternaryOperator qop = (QuaternaryOperator) _optr;

    MatrixObject matrix_object_1 = execution_context.getMatrixObject(input1);
    MatrixObject matrix_object_2 = execution_context.getMatrixObject(input2);
    MatrixObject matrix_object_3 = execution_context.getMatrixObject(input3);
    MatrixObject matrix_object_4 = null;
    if(qop.hasFourInputs())
    {
      matrix_object_4 = execution_context.getMatrixObject(_input4);
    }

    if(matrix_object_1.isFederated() && !matrix_object_2.isFederated() && !matrix_object_3.isFederated() && (matrix_object_4 == null || !matrix_object_4.isFederated()))
    {
      FederationMap federation_mapping = matrix_object_1.getFedMapping();
      FederatedRequest[] fed_req_init_1 = federation_mapping.broadcastSliced(matrix_object_2, false);
      FederatedRequest fed_req_init_2 = federation_mapping.broadcast(matrix_object_3);
      FederatedRequest fed_req_init_3 = null;
      FederatedRequest fed_req_compute_1 = null;

      if(matrix_object_4 != null)
      {
        fed_req_init_3 = federation_mapping.broadcast(matrix_object_4);
        fed_req_compute_1 = FederationUtils.callInstruction(instString, output,
          new CPOperand[]{input1, input2, input3, _input4},
          new long[]{federation_mapping.getID(), fed_req_init_1[0].getID(), fed_req_init_2.getID(), fed_req_init_3.getID()});
      }
      else
      {
        fed_req_compute_1 = FederationUtils.callInstruction(instString, output,
          new CPOperand[]{input1, input2, input3},
          new long[]{federation_mapping.getID(), fed_req_init_1[0].getID(), fed_req_init_2.getID()});
      }
      FederatedRequest fed_req_get_1 = new FederatedRequest(RequestType.GET_VAR, fed_req_compute_1.getID());
      FederatedRequest fed_req_cleanup_1 = federation_mapping.cleanup(getTID(), fed_req_compute_1.getID());
      FederatedRequest fed_req_cleanup_2 = federation_mapping.cleanup(getTID(), fed_req_init_1[0].getID());
      FederatedRequest fed_req_cleanup_3 = federation_mapping.cleanup(getTID(), fed_req_init_2.getID());

      Future<FederatedResponse>[] response;
      if(fed_req_init_3 != null)
      {
        FederatedRequest fed_req_cleanup_4 = federation_mapping.cleanup(getTID(), fed_req_init_3.getID());
        // execute federated instructions
        response = federation_mapping.execute(getTID(), true, fed_req_init_1, fed_req_init_2, fed_req_init_3,
          fed_req_compute_1, fed_req_get_1,
          fed_req_cleanup_1, fed_req_cleanup_2, fed_req_cleanup_3, fed_req_cleanup_4);
      }
      else
      {
        // execute federated instructions
        response = federation_mapping.execute(getTID(), true, fed_req_init_1, fed_req_init_2,
          fed_req_compute_1, fed_req_get_1,
          fed_req_cleanup_1, fed_req_cleanup_2, fed_req_cleanup_3);
      }

      AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");

      execution_context.setVariable(output.getName(), FederationUtils.aggScalar(aop, response));
    }
  }

}
