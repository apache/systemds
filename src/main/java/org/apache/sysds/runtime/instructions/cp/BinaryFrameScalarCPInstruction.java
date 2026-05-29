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

package org.apache.sysds.runtime.instructions.cp;

import java.util.Arrays;

import org.apache.sysds.common.Builtins;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.MultiThreadedOperator;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONObject;

public class BinaryFrameScalarCPInstruction extends BinaryCPInstruction {
	// private static final Log LOG = LogFactory.getLog(BinaryFrameFrameCPInstruction.class.getName());

	protected BinaryFrameScalarCPInstruction(MultiThreadedOperator op, CPOperand in1, CPOperand in2, CPOperand out,
		String opcode, String istr) {
		super(CPType.Binary, op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// get input frames
		FrameBlock inBlock1 = ec.getFrameInput(input1.getName());
		ScalarObject spec = ec.getScalarInput(input2.getName(), ValueType.STRING, true);
		if(getOpcode().equals(Builtins.GET_CATEGORICAL_MASK.toString().toLowerCase())) {
			processGetCategorical(ec, inBlock1, spec);
		}
		else {
			throw new DMLRuntimeException("Unsupported operation");
		}

		// Release the memory occupied by input frames
		ec.releaseFrameInput(input1.getName());
	}

	public void processGetCategorical(ExecutionContext ec, FrameBlock f, ScalarObject spec) {
		try {

			// MatrixBlock ret = new MatrixBlock();
			int nCol = f.getNumColumns();

			JSONObject jSpec = new JSONObject(spec.getStringValue());

			if(!jSpec.containsKey("ids") && jSpec.getBoolean("ids")) {
				throw new DMLRuntimeException("not supported non ID based spec for get_categorical_mask");
			}

			String recode = TfMethod.RECODE.toString();
			String dummycode = TfMethod.DUMMYCODE.toString();

			int[] lengths = new int[nCol];
			// assume all columns encode to at least one column.
			Arrays.fill(lengths, 1);
			boolean[] categorical = new boolean[nCol];

			if(jSpec.containsKey(recode)) {
				JSONArray a = jSpec.getJSONArray(recode);
				for(Object aa : a) {
					int av = (Integer) aa - 1;
					categorical[av] = true;
				}
			}

			if(jSpec.containsKey(dummycode)) {
				JSONArray a = jSpec.getJSONArray(dummycode);
				for(Object aa : a) {
					int av = (Integer) aa - 1;
					ColumnMetadata d = f.getColumnMetadata()[av];
					String v = f.getString(0, av);
					int ndist;
					if(v.length() > 1 && v.charAt(0) == '¿') {
						ndist = UtilFunctions.parseToInt(v.substring(1));
					}
					else {
						ndist = d.isDefault() ? 0 : (int) d.getNumDistinct();
					}
					lengths[av] = ndist;
					categorical[av] = true;
				}
			}

			// get total size after mapping

			int sumLengths = 0;
			for(int i : lengths) {
				sumLengths += i;
			}

			MatrixBlock ret = new MatrixBlock(1, sumLengths, false);
			ret.allocateDenseBlock();
			int off = 0;
			for(int i = 0; i < lengths.length; i++) {
				for(int j = 0; j < lengths[i]; j++) {
					ret.set(0, off++, categorical[i] ? 1 : 0);
				}
			}

			ec.setMatrixOutput(output.getName(), ret);

		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}
}
