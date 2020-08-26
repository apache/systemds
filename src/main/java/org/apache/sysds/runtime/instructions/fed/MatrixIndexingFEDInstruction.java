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

import org.apache.sysds.lops.Federated;
import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.parser.IndexedIdentifier;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.*;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.Arrays;
import java.util.concurrent.Future;

public final class MatrixIndexingFEDInstruction extends IndexingFEDInstruction {
    public MatrixIndexingFEDInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu,
                                       CPOperand out, String opcode, String istr) {
        super(in, rl, ru, cl, cu, out, opcode, istr);
    }

    //for left indexing
    protected MatrixIndexingFEDInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl,
                                          CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr) {
        super(lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
    }

    @Override
    public void processInstruction(ExecutionContext ec) {

        String opcode = getOpcode();
        IndexRange ixrange = getIndexRange(ec);

        //get original matrix
        MatrixObject mo = ec.getMatrixObject(input1.getName());
        DataCharacteristics dc = mo.getDataCharacteristics();

        if (mo.getNumRows()-1 < ixrange.rowEnd || mo.getNumColumns()-1 < ixrange.colEnd ||
                ixrange.rowStart < 0 || ixrange.colStart < 0 ||
                ixrange.rowStart > ixrange.rowEnd || ixrange.colStart > ixrange.colEnd)
            throw new DMLRuntimeException("Federated Matrix Indexing: Invalid indices.");

        //right indexing
        if(mo.isFederated() && opcode.equalsIgnoreCase(RightIndex.OPCODE)) {
            FederatedRange[] fr = mo.getFedMapping().getFederatedRanges();

            if (mo.getNumRows()-1 == ixrange.rowEnd && mo.getNumColumns()-1 == ixrange.colEnd &&
                    ixrange.rowStart == ixrange.colStart && ixrange.colStart == 0) {

//                MatrixObject out = ec.getMatrixObject(output);
//                out.getDataCharacteristics().set(mo.getDataCharacteristics());
//                out.setFedMapping(mo.getFedMapping().copyWithNewID(FederationUtils.getNextFedDataID()));

                MatrixObject out = ec.getMatrixObject(output);
                out.getDataCharacteristics().set(mo.getDataCharacteristics());
                FederatedRequest federatedRequest = FederationUtils.callInstruction(instString, output, new CPOperand[] {input1}, new long[]{mo.getFedMapping().getID()});
                mo.getFedMapping().execute(getTID(), true, federatedRequest);
                out.setFedMapping(mo.getFedMapping().copyWithNewID(federatedRequest.getID()));

//            } else if (fr.length == 1) {
//                FederatedRequest fRequest = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1}, new long[]{mo.getFedMapping().getID()});
//                mo.getFedMapping().execute(getTID(), true, fRequest);
//
//                Arrays.toString(InstructionUtils.getInstructionParts(instString));
//
//                MatrixObject out = ec.getMatrixObject(output);
//                out.getDataCharacteristics().set(ixrange.rowSpan(), ixrange.colSpan(), dc.getBlocksize(), dc.getNonZeros());
//                out.setFedMapping(mo.getFedMapping().copyWithNewID(fRequest.getID()));
//            } else if (mo.isFederated(FederationMap.FType.ROW)){
//                for (FederatedRange f : fr) {
//                    if (f.getBeginDims()[0] <= ixrange.rowStart && ixrange.rowEnd  <= f.getEndDims()[0] &&
//                            f.getBeginDims()[1] <= ixrange.colStart && ixrange.colEnd  <= f.getEndDims()[1]) {
//
//                    }
//                }

//            } else if (mo.isFederated(FederationMap.FType.COL)){
//
//
            } else throw new DMLRuntimeException("Federated Matrix Indexing: Unsupported input fed type");

        } else throw new DMLRuntimeException("Federated Matrix Indexing: "
                + "Federated input expected, but invoked w/ " + mo.isFederated());
    }
}
