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

import java.util.HashSet;
import java.util.Set;

import org.apache.sysds.common.Types;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;

public class CSVReblockFEDInstruction extends UnaryFEDInstruction {
	private int _blen;
	private boolean _hasHeader;
	private String _delim;
	private boolean _fill;
	private double _fillValue;
	private Set<String> _naStrings;

	protected CSVReblockFEDInstruction(Operator op, CPOperand in, CPOperand out, int br, int bc, boolean hasHeader,
		String delim, boolean fill, double fillValue, String opcode, String instr, Set<String> naStrings) {
		super(FEDType.CSVReblock, op, in, out, opcode, instr);
		_blen = br;
		_blen = bc;
		_hasHeader = hasHeader;
		_delim = delim;
		_fill = fill;
		_fillValue = fillValue;
		_naStrings = naStrings;
	}

	public static CSVReblockFEDInstruction parseInstruction(String str) {
		String opcode = InstructionUtils.getOpCode(str);
		if( !opcode.equals("csvrblk") )
			throw new DMLRuntimeException("Incorrect opcode for CSVReblockSPInstruction:" + opcode);

		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);

		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int blen = Integer.parseInt(parts[3]);
		boolean hasHeader = Boolean.parseBoolean(parts[4]);
		String delim = parts[5];
		boolean fill = Boolean.parseBoolean(parts[6]);
		double fillValue = Double.parseDouble(parts[7]);
		Set<String> naStrings = null;

		String[] naS = parts[8].split(DataExpression.DELIM_NA_STRING_SEP);

		if(naS.length > 0  && !(naS.length ==1 && naS[0].isEmpty())){
			naStrings = new HashSet<>();
			for(String s: naS)
				naStrings.add(s);
		}

		return new CSVReblockFEDInstruction(null, in, out, blen, blen,
			hasHeader, delim, fill, fillValue, opcode, str, naStrings);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		//set the output characteristics
		CacheableData<?> obj = ec.getCacheableData(input1.getName());
		DataCharacteristics mc = ec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = ec.getDataCharacteristics(output.getName());
		mcOut.set(mc.getRows(), mc.getCols(), _blen);

		//get the source format from the meta data
		MetaDataFormat iimd = (MetaDataFormat) obj.getMetaData();
		if (iimd.getFileFormat() != Types.FileFormat.CSV) {
			throw new DMLRuntimeException("The given format is not implemented for "
				+ "CSVReblockFEDInstruction:" + iimd.getFileFormat().toString());
		}

		long id = FederationUtils.getNextFedDataID();
		FederatedRequest fr1 = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id, mcOut, obj.getDataType());
		FederatedRequest fr2 = FederationUtils.callInstruction(instString, output, id,
			new CPOperand[]{input1}, new long[]{ obj.getFedMapping().getID()}, Types.ExecType.SPARK, false);

		//execute federated operations and set output
		obj.getFedMapping().execute(getTID(), true, fr1, fr2);
		CacheableData<?> out = ec.getMatrixObject(output);
		out.setFedMapping(obj.getFedMapping().copyWithNewID(fr2.getID()));
		out.getDataCharacteristics().set(mcOut);
	}
}
