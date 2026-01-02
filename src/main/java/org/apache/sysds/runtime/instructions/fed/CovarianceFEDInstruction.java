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
import java.util.stream.IntStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.federated.MatrixLineagePair;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.CovarianceCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.CovarianceSPInstruction;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.COVOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class CovarianceFEDInstruction extends BinaryFEDInstruction {
	
	private CovarianceFEDInstruction(Operator op, CPOperand in1,
		CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr)
	{
		super(FEDInstruction.FEDType.AggregateBinary, op, in1, in2, in3, out, opcode, istr);
	}

	public static CovarianceFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		FederatedOutput fedOut = FederatedOutput.valueOf(parts[parts.length-1]);
		String cleanInstStr = InstructionUtils.removeFEDOutputFlag(str);
		CovarianceFEDInstruction fedInst = parseInstruction(CovarianceCPInstruction.parseInstruction(cleanInstStr));
		fedInst._fedOut = fedOut;
		return fedInst;
	}

	public static CovarianceFEDInstruction parseInstruction(CovarianceCPInstruction inst) {
		return new CovarianceFEDInstruction(inst.getOperator(), inst.input1, inst.input2, inst.input3, inst.output,
			inst.getOpcode(), inst.getInstructionString());
	}

	public static CovarianceFEDInstruction parseInstruction(CovarianceSPInstruction inst) {
		return new CovarianceFEDInstruction(inst.getOperator(), inst.input1, inst.input2, inst.input3, inst.output,
			inst.getOpcode(), inst.getInstructionString());
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		MatrixObject mo2 = ec.getMatrixObject(input2);
		MatrixLineagePair weights = (input3 != null) ? ec.getMatrixLineagePair(input3) : null;

		if(mo1.isFederated() && mo2.isFederated() && !mo1.getFedMapping().isAligned(mo2.getFedMapping(), false))
			throw new DMLRuntimeException("Not supported matrix-matrix binary operation: covariance.");

		boolean moAligned = mo1.isFederated() && mo2.isFederated() && mo1.getFedMapping().isAligned(mo2.getFedMapping(), false);
		boolean weightsAligned = weights == null || (weights.isFederated() && mo2.isFederated()
			&& weights.getFedMapping().isAligned(mo2.getFedMapping(), false));

		// all aligned
		if(moAligned && weightsAligned)
			processAlignedFedCov(ec, mo1, mo2, weights);
		// weights are not aligned, broadcast
		else if(moAligned)
			processFedCovWeights(ec, mo1, mo2, weights);
		else
			processCov(ec, mo1, mo2);
	}

	private void processAlignedFedCov(ExecutionContext ec, MatrixObject mo1, MatrixObject mo2,
		MatrixLineagePair moLin3) {
		FederatedRequest fr1;
		if(moLin3 == null) {
			fr1 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2}, new long[]{mo1.getFedMapping().getID(), mo2.getFedMapping().getID()});
		}
		else {
			fr1 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2, input3}, new long[]{mo1.getFedMapping().getID(),
					mo2.getFedMapping().getID(), moLin3.getFedMapping().getID()});
		}

		FederatedRequest fr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr1.getID());
		FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());
		Double[] cov = getResponses(mo1.getFedMapping().execute(getTID(), fr1, fr2, fr3));
		Double[] mean1 = getResponses(processMean(mo1, moLin3, 0));
		Double[] mean2 = getResponses(processMean(mo2, moLin3, 1));

		if (moLin3 == null) {
			double result = aggCov(cov, mean1, mean2, mo1.getFedMapping().getFederatedRanges());
			ec.setVariable(output.getName(), new DoubleObject(result));
		}
		else {
			Double[] weights = getResponses(
				getWeightsSum(moLin3, moLin3.getFedMapping().getID(), instString, moLin3.getFedMapping()));
			double result = aggWeightedCov(cov, mean1, mean2, weights);
			ec.setVariable(output.getName(), new DoubleObject(result));
		}
	}

	private void processFedCovWeights(ExecutionContext ec, MatrixObject mo1, MatrixObject mo2,
		MatrixLineagePair moLin3) {
		
		FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(moLin3, false);

		// the original instruction encodes weights as "pREADW", change to the new ID
		String[] parts = instString.split(Lop.OPERAND_DELIMITOR);
		String covInstr = instString.replace(parts[4], String.valueOf(fr1[0].getID()) + "·MATRIX·FP64");

		FederatedRequest fr2 = FederationUtils.callInstruction(
			covInstr, output,
			new CPOperand[]{input1, input2, input3},
			new long[]{mo1.getFedMapping().getID(), mo2.getFedMapping().getID(), fr1[0].getID()}
		);
		//sequential execution of cov and means for robustness
		FederatedRequest fr3 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, fr2.getID());
		FederatedRequest fr4 = mo1.getFedMapping().cleanup(getTID(), fr2.getID());
		Double[] cov = getResponses(mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3, fr4));
		Double[] mean1 = getResponses(processMean(mo1, 0, fr1[0].getID()));
		Double[] mean2 = getResponses(processMean(mo2, 1, fr1[0].getID()));
		Double[] weights = getResponses(getWeightsSum(moLin3, fr1[0].getID(), instString, mo1.getFedMapping()));
		double result = aggWeightedCov(cov, mean1, mean2, weights);
		ec.setVariable(output.getName(), new DoubleObject(result));
	}

	private void processCov(ExecutionContext ec, MatrixObject mo1, MatrixObject mo2) {
		MatrixBlock mb;
		MatrixObject mo;
		COVOperator cop = ((COVOperator)_optr);

		if(!mo1.isFederated() && mo2.isFederated()) {
			mo = mo2;
			mb = ec.getMatrixInput(input1.getName());
		}
		else {
			mo = mo1;
			mb = ec.getMatrixInput(input2.getName());
		}

		FederationMap fedMapping = mo.getFedMapping();
		List<CmCovObject> globalCmobj = new ArrayList<>();
		long varID = FederationUtils.getNextFedDataID();
		fedMapping.mapParallel(varID, (range, data) -> {

			FederatedResponse response;
			try {
				if(input3 == null) {
					response = data.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
						new CovarianceFEDInstruction.COVFunction(data.getVarID(),
							mb.slice(range.getBeginDimsInt()[0], range.getEndDimsInt()[0] - 1),
							cop))).get();
				}
				// with weights
				else {
					MatrixBlock wtBlock = ec.getMatrixInput(input3.getName());
					response = data.executeFederatedOperation(
						new FederatedRequest(
							FederatedRequest.RequestType.EXEC_UDF, -1,
							new CovarianceFEDInstruction.COVWeightsFunction(
								data.getVarID(),
								mb.slice(range.getBeginDimsInt()[0], range.getEndDimsInt()[0] - 1),
								cop, wtBlock.slice(range.getBeginDimsInt()[0], range.getEndDimsInt()[0] - 1)
							)
						)
					).get();
				}

				if(!response.isSuccessful())
					response.throwExceptionFromResponse();
				synchronized(globalCmobj) {
					globalCmobj.add((CmCovObject) response.getData()[0]);
				}
			}
			catch(Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});

		Optional<CmCovObject> res = globalCmobj.stream().reduce((arg0, arg1) -> (CmCovObject) cop.fn.execute(arg0, arg1));
		try {
			ec.setScalarOutput(output.getName(), new DoubleObject(res.get().getRequiredResult(cop)));
		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}

	private static Double[] getResponses(Future<FederatedResponse>[] ffr) {
		Double[] fr = new Double[ffr.length];
		IntStream.range(0, fr.length).forEach(i -> {
			try {
				fr[i] = ((ScalarObject) ffr[i].get().getData()[0]).getDoubleValue();
			}
			catch(Exception e) {
				throw new DMLRuntimeException("CovarianceFEDInstruction: incorrect means or cov.", e);
			}
		});

		return fr;
	}

	private static double aggCov(Double[] covValues, Double[] mean1, Double[] mean2, FederatedRange[] ranges) {
		long[] sizes = new long[ranges.length];
		for (int i = 0; i < ranges.length; i++) {
			sizes[i] = ranges[i].getSize();
		}
		
		// calculate global means
		double totalMeanX = 0;
		double totalMeanY = 0;
		int totalCount = 0;
		for (int i = 0; i < mean1.length; i++) {
			totalMeanX += mean1[i] * sizes[i];
			totalMeanY += mean2[i] * sizes[i];
			totalCount += sizes[i];
		}

		totalMeanX /= totalCount;
		totalMeanY /= totalCount;

		// calculate global covariance
		double cov = 0;
		for (int i = 0; i < covValues.length; i++) {
			cov += (sizes[i] - 1) * covValues[i];
			cov += sizes[i] * (mean1[i] - totalMeanX) * (mean2[i] - totalMeanY);
		}
		return cov / (totalCount - 1); // adjusting for degrees of freedom
	}

	private static double aggWeightedCov(Double[] covValues, Double[] mean1, Double[] mean2, Double[] weights) {
		// calculate global weighted means
		double totalWeightedMeanX = 0;
		double totalWeightedMeanY = 0;
		double totalWeight = 0;
		for (int i = 0; i < mean1.length; i++) {
			totalWeight += weights[i];
			totalWeightedMeanX += mean1[i] * weights[i];
			totalWeightedMeanY += mean2[i] * weights[i];
		}

		totalWeightedMeanX /= totalWeight;
		totalWeightedMeanY /= totalWeight;

		// calculate global weighted covariance
		double cov = 0;
		for (int i = 0; i < covValues.length; i++) {
			cov += (weights[i] - 1) * covValues[i];
			cov += weights[i] * (mean1[i] - totalWeightedMeanX) * (mean2[i] - totalWeightedMeanY);
		}
		return cov / (totalWeight - 1); // adjusting for degrees of freedom
	}

	private Future<FederatedResponse>[] processMean(MatrixObject mo1, MatrixLineagePair moLin3, int var){
		String[] parts = instString.split(Lop.OPERAND_DELIMITOR);
		Future<FederatedResponse>[] meanTmp = null;
		if (moLin3 == null) {
			String meanInstr = instString.replace(getOpcode(), getOpcode().replace("cov", "uamean"));
			meanInstr = meanInstr.replace((var == 0 ? parts[2] : parts[3]) + Lop.OPERAND_DELIMITOR, "");
			meanInstr = meanInstr.replace(parts[4], parts[4].replace("FP64", "STRING°16"));

			//create federated commands for aggregation
			FederatedRequest meanFr1 = FederationUtils.callInstruction(meanInstr, output,
				new CPOperand[]{var == 0 ? input2 : input1}, new long[]{mo1.getFedMapping().getID()});
			FederatedRequest meanFr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, meanFr1.getID());
			FederatedRequest meanFr3 = mo1.getFedMapping().cleanup(getTID(), meanFr1.getID());

			meanTmp = mo1.getFedMapping().execute(getTID(), meanFr1, meanFr2, meanFr3);
		}
		else {
			// multiply input X by weights W element-wise
			String multOutput = incrementVar(parts[4], 1);
			String multInstr = instString
				.replace(getOpcode(), getOpcode().replace("cov", "*"))
				.replace((var == 0 ? parts[2] : parts[3]) + Lop.OPERAND_DELIMITOR, "")
				.replace(parts[5], multOutput);

			CPOperand multOutputCPOp = new CPOperand(
				multOutput.substring(0, multOutput.indexOf("·")),
				mo1.getValueType(), mo1.getDataType()
			);

			FederatedRequest multFr = FederationUtils.callInstruction(
				multInstr,
				multOutputCPOp,
				new CPOperand[]{var == 0 ? input2 : input1, input3},
				new long[]{mo1.getFedMapping().getID(), moLin3.getFedMapping().getID()}
			);

			// calculate the sum of the obtained vector
			String[] partsMult = multInstr.split(Lop.OPERAND_DELIMITOR);
			String sumInstr1Output = incrementVar(multOutput, 1)
				.replace("m", "")
				.replace("MATRIX", "SCALAR");
			String sumInstr1 = multInstr
				.replace(partsMult[1], "uak+")
				.replace(partsMult[3] + Lop.OPERAND_DELIMITOR, "")
				.replace(partsMult[4], sumInstr1Output)
				.replace(partsMult[2], multOutput);

			FederatedRequest sumFr1 = FederationUtils.callInstruction(
				sumInstr1,
				new CPOperand(
					sumInstr1Output.substring(0, sumInstr1Output.indexOf("·")),
					output.getValueType(), output.getDataType()
				),
				new CPOperand[]{multOutputCPOp},
				new long[]{multFr.getID()}
			);

			// calculate the sum of weights
			String[] partsSum1 = sumInstr1.split(Lop.OPERAND_DELIMITOR);
			String sumInstr2Output = incrementVar(sumInstr1Output, 1);
			String sumInstr2 = sumInstr1
				.replace(partsSum1[2], parts[4])
				.replace(partsSum1[3], sumInstr2Output);

			FederatedRequest sumFr2 = FederationUtils.callInstruction(
				sumInstr2,
				new CPOperand(
					sumInstr2Output.substring(0, sumInstr2Output.indexOf(Lop.DATATYPE_PREFIX)),
					output.getValueType(), output.getDataType()
				),
				new CPOperand[]{input3},
				new long[]{moLin3.getFedMapping().getID()}
			);

			// divide sum(X*W) by sum(W)
			String[] partsSum2 = sumInstr2.split(Lop.OPERAND_DELIMITOR);
			String divInstrOutput = incrementVar(sumInstr2Output, 1);
			String divInstrInput1 = partsSum2[2].replace(partsSum2[2], sumInstr1Output + Lop.DATATYPE_PREFIX + "false");
			String divInstrInput2 = partsSum2[3].replace(partsSum2[3], sumInstr2Output + Lop.DATATYPE_PREFIX + "false");
			String divInstr = partsSum2[0] + Lop.OPERAND_DELIMITOR + partsSum2[1].replace("uak+", "/") 
				+ Lop.OPERAND_DELIMITOR + divInstrInput1 + Lop.OPERAND_DELIMITOR + divInstrInput2 
				+ Lop.OPERAND_DELIMITOR + divInstrOutput + Lop.OPERAND_DELIMITOR + partsSum2[4];

			FederatedRequest divFr1 = FederationUtils.callInstruction(
				divInstr,
				new CPOperand(
					divInstrOutput.substring(0, divInstrOutput.indexOf("·")),
					output.getValueType(), output.getDataType()
				),
				new CPOperand[]{
					new CPOperand(
						sumInstr1Output.substring(0, sumInstr1Output.indexOf(Lop.DATATYPE_PREFIX)),
						output.getValueType(), output.getDataType(), output.isLiteral()
					),
					new CPOperand(
						sumInstr2Output.substring(0, sumInstr2Output.indexOf(Lop.DATATYPE_PREFIX)),
						output.getValueType(), output.getDataType(), output.isLiteral()
					)
				},
				new long[]{sumFr1.getID(), sumFr2.getID()}
			);
			FederatedRequest divFr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, divFr1.getID());
			FederatedRequest divFr3 = mo1.getFedMapping().cleanup(getTID(), multFr.getID(), sumFr1.getID(), sumFr2.getID(), divFr1.getID(), divFr2.getID());

			meanTmp = mo1.getFedMapping().execute(getTID(), multFr, sumFr1, sumFr2, divFr1, divFr2, divFr3);
		}
		return meanTmp;
	}

	private Future<FederatedResponse>[] processMean(MatrixObject mo1, int var, long weightsID){
		String[] parts = instString.split(Lop.OPERAND_DELIMITOR);
		Future<FederatedResponse>[] meanTmp = null;

		// multiply input X by weights W element-wise
		String multOutput = (var == 0 ? incrementVar(parts[2], 5) : incrementVar(parts[3], 3));
		String multInstr = instString
			.replace(getOpcode(), getOpcode().replace("cov", "*"))
			.replace((var == 0 ? parts[2] : parts[3]) + Lop.OPERAND_DELIMITOR, "")
			.replace(parts[4], String.valueOf(weightsID) + "·MATRIX·FP64")
			.replace(parts[5], multOutput);

		CPOperand multOutputCPOp = new CPOperand(
			multOutput.substring(0, multOutput.indexOf(Lop.DATATYPE_PREFIX)),
			mo1.getValueType(), mo1.getDataType()
		);

		FederatedRequest multFr = FederationUtils.callInstruction(
			multInstr,
			multOutputCPOp,
			new CPOperand[]{var == 0 ? input2 : input1, input3},
			new long[]{mo1.getFedMapping().getID(), weightsID}
		);

		// calculate the sum of the obtained vector
		String[] partsMult = multInstr.split(Lop.OPERAND_DELIMITOR);
		String sumInstr1Output = incrementVar(multOutput, 1)
			.replace("m", "")
			.replace("MATRIX", "SCALAR");
		String sumInstr1 = multInstr
			.replace(partsMult[1], "uak+")
			.replace(partsMult[3] + Lop.OPERAND_DELIMITOR, "")
			.replace(partsMult[4], sumInstr1Output)
			.replace(partsMult[2], multOutput);

		FederatedRequest sumFr1 = FederationUtils.callInstruction(
			sumInstr1,
			new CPOperand(
				sumInstr1Output.substring(0, sumInstr1Output.indexOf(Lop.DATATYPE_PREFIX)),
				output.getValueType(), output.getDataType()
			),
			new CPOperand[]{multOutputCPOp},
			new long[]{multFr.getID()}
		);

		// calculate the sum of weights
		String[] partsSum1 = sumInstr1.split(Lop.OPERAND_DELIMITOR);
		String sumInstr2Output = incrementVar(sumInstr1Output, 1);
		String sumInstr2 = sumInstr1
			.replace(partsSum1[2], String.valueOf(weightsID) + "·MATRIX·FP64")
			.replace(partsSum1[3], sumInstr2Output);

		FederatedRequest sumFr2 = FederationUtils.callInstruction(
			sumInstr2,
			new CPOperand(
				sumInstr2Output.substring(0, sumInstr2Output.indexOf(Lop.DATATYPE_PREFIX)),
				output.getValueType(), output.getDataType()
			),
			new CPOperand[]{input3},
			new long[]{weightsID}
		);

		// divide sum(X*W) by sum(W)
		String[] partsSum2 = sumInstr2.split(Lop.OPERAND_DELIMITOR);
		String divInstrOutput = incrementVar(sumInstr2Output, 1);
		String divInstrInput1 = partsSum2[2].replace(partsSum2[2], sumInstr1Output + Lop.DATATYPE_PREFIX + "false");
		String divInstrInput2 = partsSum2[3].replace(partsSum2[3], sumInstr2Output + Lop.DATATYPE_PREFIX + "false");
		String divInstr = partsSum2[0] + Lop.OPERAND_DELIMITOR + partsSum2[1].replace("uak+", "/") + Lop.OPERAND_DELIMITOR 
				+ divInstrInput1 + Lop.OPERAND_DELIMITOR + divInstrInput2 + Lop.OPERAND_DELIMITOR 
				+ divInstrOutput + Lop.OPERAND_DELIMITOR + partsSum2[4];

		FederatedRequest divFr1 = FederationUtils.callInstruction(
			divInstr,
			new CPOperand(
				divInstrOutput.substring(0, divInstrOutput.indexOf(Lop.DATATYPE_PREFIX)),
				output.getValueType(), output.getDataType()
			),
			new CPOperand[]{
				new CPOperand(
					sumInstr1Output.substring(0, sumInstr1Output.indexOf(Lop.DATATYPE_PREFIX)),
					output.getValueType(), output.getDataType(), output.isLiteral()
				),
				new CPOperand(
					sumInstr2Output.substring(0, sumInstr2Output.indexOf(Lop.DATATYPE_PREFIX)),
					output.getValueType(), output.getDataType(), output.isLiteral()
				)
			},
			new long[]{sumFr1.getID(), sumFr2.getID()}
		);
		FederatedRequest divFr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, divFr1.getID());
		FederatedRequest divFr3 = mo1.getFedMapping().cleanup(getTID(), multFr.getID(), sumFr1.getID(), sumFr2.getID(), divFr1.getID(), divFr2.getID());

		meanTmp = mo1.getFedMapping().execute(getTID(), multFr, sumFr1, sumFr2, divFr1, divFr2, divFr3);
		return meanTmp;
	}

	private Future<FederatedResponse>[] getWeightsSum(MatrixLineagePair moLin3, long weightsID, String instString, FederationMap fedMap) {
		Future<FederatedResponse>[] weightsSumTmp = null;

		String[] parts = instString.split(Lop.OPERAND_DELIMITOR);
		if (!instString.contains("pREADW")) {
			String sumInstr = "CP"+Lop.OPERAND_DELIMITOR+"uak+" + Lop.OPERAND_DELIMITOR 
				+ parts[4] + Lop.OPERAND_DELIMITOR + parts[5] + Lop.OPERAND_DELIMITOR + parts[6];

			FederatedRequest sumFr = FederationUtils.callInstruction(
				sumInstr,
				new CPOperand(
					parts[5].substring(0, parts[5].indexOf(Lop.DATATYPE_PREFIX)),
					output.getValueType(),
					output.getDataType()
				),
				new CPOperand[]{input3},
				new long[]{weightsID}
			);
			FederatedRequest sumFr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, sumFr.getID());
			FederatedRequest sumFr3 = moLin3.getFedMapping().cleanup(getTID(), sumFr.getID());

			weightsSumTmp = fedMap.execute(getTID(), sumFr, sumFr2, sumFr3);
		}
		else {
			String sumInstr = "CP"+Lop.OPERAND_DELIMITOR+"uak+"+Lop.OPERAND_DELIMITOR
				+ String.valueOf(weightsID) + "·MATRIX·FP64" + Lop.OPERAND_DELIMITOR + parts[5] 
				+ Lop.OPERAND_DELIMITOR + parts[6];
			FederatedRequest sumFr = FederationUtils.callInstruction(
				sumInstr,
				new CPOperand(
					parts[5].substring(0, parts[5].indexOf(Lop.DATATYPE_PREFIX)),
					output.getValueType(),
					output.getDataType()
				),
				new CPOperand[]{input3},
				new long[]{weightsID}
			);
			FederatedRequest sumFr2 = new FederatedRequest(FederatedRequest.RequestType.GET_VAR, sumFr.getID());
			FederatedRequest sumFr3 = fedMap.cleanup(getTID(), sumFr.getID());

			weightsSumTmp = fedMap.execute(getTID(), sumFr, sumFr2, sumFr3);
		}
		return weightsSumTmp;
	}

	private static String incrementVar(String str, int inc) {
		StringBuilder strOut = new StringBuilder(str);
		Pattern pattern = Pattern.compile("\\d+");
		Matcher matcher = pattern.matcher(strOut);
		if (matcher.find()) {
			int num = Integer.parseInt(matcher.group()) + inc;
			int start = matcher.start();
			int end = matcher.end();
			strOut.replace(start, end, String.valueOf(num));
		}
		return strOut.toString();
	}

	private static class COVFunction extends FederatedUDF {

		private static final long serialVersionUID = -501036588060113499L;
		private final MatrixBlock _mo2;
		private final COVOperator _op;

		public COVFunction (long input, MatrixBlock mo2, COVOperator op) {
			super(new long[] {input});
			_op = op;
			_mo2 = mo2;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, mb.covOperations(_op, _mo2));
		}

		@Override 
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	private static class COVWeightsFunction extends FederatedUDF {
		private static final long serialVersionUID = -1768739786192949573L;
		private final COVOperator _op;
		private final MatrixBlock _mo2;
		private final MatrixBlock _weights;

		protected COVWeightsFunction(long input, MatrixBlock mo2, COVOperator op, MatrixBlock weights) {
			super(new long[] {input});
			_mo2 = mo2;
			_op = op;
			_weights = weights;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, mb.covOperations(_op, _mo2, _weights));
		}

		@Override 
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}
}
