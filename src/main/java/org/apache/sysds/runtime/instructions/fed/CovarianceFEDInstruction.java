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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Future;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.functionobjects.COV;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.CovarianceCPInstruction;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.COVOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

public class CovarianceFEDInstruction extends BinaryFEDInstruction {
	private CovarianceFEDInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
		String istr) {
		super(FEDInstruction.FEDType.AggregateBinary, op, in1, in2, out, opcode, istr);
	}

	private CovarianceFEDInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		String opcode, String istr) {
		super(FEDInstruction.FEDType.AggregateBinary, op, in1, in2, in3, out, opcode, istr);
	}


	public static CovarianceFEDInstruction parseInstruction(String str) {
		CPOperand in1 = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);
		CPOperand in3 = null;
		CPOperand out = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if( !opcode.equalsIgnoreCase("cov") ) {
			throw new DMLRuntimeException("CovarianceCPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}

		COVOperator cov = new COVOperator(COV.getCOMFnObject());
		if ( parts.length == 4 ) {
			// CP.cov.mVar0.mVar1.mVar2
			parseBinaryInstruction(str, in1, in2, out);
			return new CovarianceFEDInstruction(cov, in1, in2, out, opcode, str);
		} else if ( parts.length == 5 ) {
			// CP.cov.mVar0.mVar1.mVar2.mVar3
			in3 = new CPOperand("", Types.ValueType.UNKNOWN, Types.DataType.UNKNOWN);
			parseBinaryInstruction(str, in1, in2, in3, out);
			return new CovarianceFEDInstruction(cov, in1, in2, in3, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Invalid number of arguments in Instruction: " + str);
		}
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		MatrixObject mo2 = ec.getMatrixObject(input2);

		//cov
		FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
		FederatedRequest fr2 = FederationUtils.callInstruction(instString, output,
			new CPOperand[]{input1, input2}, new long[]{mo1.getFedMapping().getID(), fr1[0].getID()});
		FederatedRequest fr3 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr2.getID());
		FederatedRequest fr4 = mo1.getFedMapping()
			.cleanup(getTID(), fr1[0].getID(), fr2.getID());

		//execute federated operations and aggregate
		Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), fr1, fr2, fr3, fr4);

//		Future<FederatedResponse>[] mult = processMult(mo1, mo2);

		//means
		Future<FederatedResponse>[] meanTmp1 = processMean(mo1, 0);
		Future<FederatedResponse>[] meanTmp2 = processMean(mo2, 1);

		double res = aggCov(tmp, meanTmp1, meanTmp2, mo1.getFedMapping());
		ec.setVariable(output.getName(), new DoubleObject(res));
	}

	private static double aggCov(Future<FederatedResponse>[] ffr, Future<FederatedResponse>[] mean1Ffr, Future<FederatedResponse>[] mean2Ffr, FederationMap map) {
		try {
			FederatedRange[] ranges = map.getFederatedRanges();
			double cov = ((ScalarObject)ffr[0].get().getData()[0]).getDoubleValue();
			double mean1 = ((ScalarObject)mean1Ffr[0].get().getData()[0]).getDoubleValue();
			double mean2 = ((ScalarObject)mean2Ffr[0].get().getData()[0]).getDoubleValue();
			double mean = (mean1 + mean2) / 2;
			long size1 = ranges[0].getSize();
			for(int i=0; i < ffr.length - 1; i++) {
				double nextCov = ((ScalarObject)ffr[i+1].get().getData()[0]).getDoubleValue();
				double nextMean1 = ((ScalarObject)mean1Ffr[i+1].get().getData()[0]).getDoubleValue();
				double nextMean2 = ((ScalarObject)mean2Ffr[i+1].get().getData()[0]).getDoubleValue();
				long size2 = ranges[i+1].getSize();

				double nextMean = (nextMean1 + nextMean2) / 2;
				double newMean = (size1 * mean + size2 * nextMean) / (size1 + size2);
				long newSize = size1 + size2;

				cov = (size1*cov + size2*nextCov + size1*(mean-newMean)*(mean-newMean) + size2*(nextMean-newMean)*(nextMean-newMean)) / newSize;

				mean = newMean;
				size1 = newSize;
			}

			return cov;

		}
		catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	private Future<FederatedResponse>[] processMean(MatrixObject mo1, int var){
		String[] parts = instString.split("°");
		String meanInstr = instString.replace(getOpcode(), getOpcode().replace("cov", "uamean"));
		meanInstr = meanInstr.replace((var == 0 ? parts[2] : parts[3]) + "°", "");
		meanInstr = meanInstr.replace(parts[4], parts[4].replace("FP64", "STRING°16"));
		Future<FederatedResponse>[] meanTmp = null;

		//create federated commands for aggregation
		FederatedRequest meanFr1 = FederationUtils.callInstruction(meanInstr, output,
			new CPOperand[]{var == 0 ? input2 : input1}, new long[]{mo1.getFedMapping().getID()});
		FederatedRequest meanFr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, meanFr1.getID());
		FederatedRequest meanFr3 = mo1.getFedMapping().cleanup(getTID(), meanFr1.getID());
		meanTmp = mo1.getFedMapping().execute(getTID(), meanFr1, meanFr2, meanFr3);
		return meanTmp;
	}

	private Future<FederatedResponse>[] processMult(MatrixObject mo1, MatrixObject mo2){
		//		CP°*°_mVar51·MATRIX·FP64°_mVar53·MATRIX·FP64°_mVar54·MATRIX·FP64
		//		CP°cov°_mVar43·MATRIX·FP64°_mVar45·MATRIX·FP64°_Var46·SCALAR·FP64

		String[] parts = instString.split("°");
		String meanInstr = instString.replace(getOpcode(), getOpcode().replace("cov", "*"));
		meanInstr = meanInstr.replace(parts[4], parts[3]);

//		FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
//		FederatedRequest fr2 = FederationUtils.callInstruction(meanInstr, input2,
//			new CPOperand[]{input1, input2}, new long[]{mo1.getFedMapping().getID(), fr1[0].getID()});
//		FederatedRequest fr3 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr2.getID());
//		FederatedRequest fr4 = mo1.getFedMapping()
//			.cleanup(getTID(), fr1[0].getID(), fr2.getID());

//		//execute federated operations and aggregate
//		Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), fr1, fr2, fr3, fr4);

		FederatedRequest fr2 = FederationUtils.callInstruction(meanInstr, input2, new CPOperand[]{input1, input2},
			new long[]{mo1.getFedMapping().getID(), mo2.getFedMapping().getID()});
		Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), true, fr2);

		return tmp;
	}

}
