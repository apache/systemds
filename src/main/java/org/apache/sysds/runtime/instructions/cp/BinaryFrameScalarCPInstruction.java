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
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class BinaryFrameScalarCPInstruction extends BinaryCPInstruction {
	// private static final Log LOG = LogFactory.getLog(BinaryFrameFrameCPInstruction.class.getName());

	private static final TfMethod[] UNSUPPORTED_MASK_METHODS = new TfMethod[] {TfMethod.BIN,
		TfMethod.WORD_EMBEDDING, TfMethod.BAG_OF_WORDS, TfMethod.UDF};

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

	private static void validate(JSONObject jSpec) {
		try {
			if(!jSpec.containsKey("ids") || !jSpec.getBoolean("ids"))
				throw new DMLRuntimeException("not supported non ID based spec for get_categorical_mask");

			for(TfMethod m : UNSUPPORTED_MASK_METHODS)
				if(jSpec.containsKey(m.toString()))
					throw new DMLRuntimeException("unsupported transform method '" + m + "' for get_categorical_mask");
		}
		catch(JSONException e) {
			throw new DMLRuntimeException(e);
		}
	}

	public void processGetCategorical(ExecutionContext ec, FrameBlock f, ScalarObject spec) {
		try {
			// 1. extract the spec, 2. validate it
			JSONObject jSpec = new JSONObject(spec.getStringValue());
			validate(jSpec);

			// 3.-5. fold each supported transform method into the per-column mask state
			CategoricalMask mask = new CategoricalMask(f, jSpec);
			mask.hash();
			mask.recode();
			mask.dummycode();

			// 6.-7. size and materialize the output mask
			ec.setMatrixOutput(output.getName(), mask.toMatrixBlock());
		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}

	/**
	 * Accumulates, per input column, how many output columns it expands to (lengths) and whether those
	 * output columns are categorical (categorical). The arrays are allocated lazily: a column that no
	 * method touches keeps the implicit default of a single, non-categorical output column.
	 */
	private static final class CategoricalMask {
		private final FrameBlock f;
		private final JSONObject jSpec;
		private final int nCol;

		private int[] lengths = null;
		private boolean[] categorical = null;

		// feature-hashed columns map to K buckets; a plain hashed column produces a single
		// (categorical) bucket-id column, while a hashed column that is additionally dummycoded
		// expands to K columns.
		private boolean[] hashed = null;
		private int K = 0;

		private CategoricalMask(FrameBlock f, JSONObject jSpec) {
			this.f = f;
			this.jSpec = jSpec;
			this.nCol = f.getNumColumns();
		}

		private void hash() throws JSONException {
			String hash = TfMethod.HASH.toString();
			if(!jSpec.containsKey(hash))
				return;
			K = jSpec.getInt("K");
			hashed = new boolean[nCol];
			ensureCategorical();
			for(Object aa : jSpec.getJSONArray(hash)) {
				int av = (Integer) aa - 1;
				hashed[av] = true;
				categorical[av] = true;
			}
		}

		private void recode() throws JSONException {
			String recode = TfMethod.RECODE.toString();
			if(!jSpec.containsKey(recode))
				return;
			ensureCategorical();
			for(Object aa : jSpec.getJSONArray(recode)) {
				int av = (Integer) aa - 1;
				categorical[av] = true;
			}
		}

		private void dummycode() throws JSONException {
			String dummycode = TfMethod.DUMMYCODE.toString();
			if(!jSpec.containsKey(dummycode))
				return;
			ensureCategorical();
			ensureLengths();
			for(Object aa : jSpec.getJSONArray(dummycode)) {
				int av = (Integer) aa - 1;
				lengths[av] = distinctCount(av);
				categorical[av] = true;
			}
		}

		private int distinctCount(int av) {
			if(hashed != null && hashed[av])
				// feature hashing followed by dummycoding yields K columns
				return K;
			ColumnMetadata d = f.getColumnMetadata()[av];
			String v = f.getString(0, av);
			if(v.length() > 1 && v.charAt(0) == '¿')
				return UtilFunctions.parseToInt(v.substring(1));
			return d.isDefault() ? 0 : (int) d.getNumDistinct();
		}

		private int sumLengths() {
			if(lengths == null)
				return nCol;
			int sum = 0;
			for(int i = 0; i < nCol; i++)
				sum += lengths[i];
			return sum;
		}

		private MatrixBlock toMatrixBlock() {
			MatrixBlock ret = new MatrixBlock(1, sumLengths(), false);
			ret.allocateDenseBlock();
			int off = 0;
			for(int i = 0; i < nCol; i++) {
				int len = (lengths == null) ? 1 : lengths[i];
				double val = (categorical != null && categorical[i]) ? 1 : 0;
				for(int j = 0; j < len; j++)
					ret.set(0, off++, val);
			}
			return ret;
		}

		private void ensureCategorical() {
			if(categorical == null)
				categorical = new boolean[nCol];
		}

		private void ensureLengths() {
			if(lengths == null) {
				lengths = new int[nCol];
				Arrays.fill(lengths, 1);
			}
		}
	}
}
