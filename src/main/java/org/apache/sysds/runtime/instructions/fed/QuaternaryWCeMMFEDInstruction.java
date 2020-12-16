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

import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
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
    // MatrixObject matrix_object_4 = execution_context.getMatrixObject(_input4);
    ScalarObject scalar_object_4 = execution_context.getScalarInput(_input4);

    if(matrix_object_1.isFederated() && !matrix_object_2.isFederated() && !matrix_object_3.isFederated())
    {
      FederationMap federation_mapping = matrix_object_1.getFedMapping();
      FederatedRequest fed_req_init_1 = federation_mapping.broadcast(matrix_object_2);
      FederatedRequest fed_req_init_2 = federation_mapping.broadcast(matrix_object_3);
      // FederatedRequest fed_req_init_3 = federation_mapping.broadcast(matrix_object_4);
      FederatedRequest fed_req_init_3 = federation_mapping.broadcast(scalar_object_4);
      FederatedRequest fed_req_compute_1 = FederationUtils.callInstruction(instString, output,
        new CPOperand[]{input1, input2, input3, _input4},
        new long[]{federation_mapping.getID(), fed_req_init_1.getID(), fed_req_init_2.getID(), fed_req_init_3.getID()});
      FederatedRequest fed_req_cleanup_1 = federation_mapping.cleanup(getTID(), fed_req_init_1.getID());
      FederatedRequest fed_req_cleanup_2 = federation_mapping.cleanup(getTID(), fed_req_init_2.getID());
      FederatedRequest fed_req_cleanup_3 = federation_mapping.cleanup(getTID(), fed_req_init_3.getID());

      // execute federated instructions
      Future<FederatedResponse>[] response = federation_mapping.execute(getTID(), true, fed_req_init_1, fed_req_init_2, fed_req_init_3, fed_req_compute_1, fed_req_cleanup_1, fed_req_cleanup_2, fed_req_cleanup_3);

      try
      {
        for(Future<FederatedResponse> tmp : response)
        {
          if(!tmp.get().isSuccessful())
          {
            tmp.get().throwExceptionFromResponse();
          }
        }
      }
      catch(Exception e)
      {
        throw new DMLRuntimeException(e);
      }

      // TODO: set the output
      // NOTE: this is only a test if setting the output would work like this
      execution_context.setVariable(output.getName(), new DoubleObject(13));

    }
  }

}