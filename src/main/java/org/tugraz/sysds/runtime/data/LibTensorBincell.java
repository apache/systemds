/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.runtime.data;

import org.tugraz.sysds.runtime.matrix.operators.BinaryOperator;
import org.tugraz.sysds.runtime.util.UtilFunctions;

public class LibTensorBincell {
	public static boolean isValidDimensionsBinary(TensorBlock m1, TensorBlock m2) {
		if (m1.getNumDims() != m2.getNumDims())
			return false;

		for (int i = 0; i < m1.getNumDims(); i++)
			if (m1.getDim(i) != m2.getDim(i) && m2.getDim(i) != 1)
				return false;
		return true;
	}

	/**
	 * tensor-tensor binary operations
	 *
	 * @param m1  input tensor 1
	 * @param m2  input tensor 2
	 * @param ret result tensor
	 * @param op  binary operator
	 */
	public static void bincellOp(BasicTensor m1, BasicTensor m2, BasicTensor ret, BinaryOperator op) {
		// TODO separate implementations for matching dims and broadcasting
		// TODO perf (empty, sparse safe, etc.)
		int[] ix1 = new int[ret.getNumDims()];
		int[] ix2 = new int[ret.getNumDims()];
		for (long i = 0; i < ret.getLength(); i++) {
			double v1 = UtilFunctions.objectToDouble(m1.getValueType(), m1.get(ix1));
			double v2 = UtilFunctions.objectToDouble(m2.getValueType(), m2.get(ix2));
			ret.set(ix1, op.fn.execute(v1, v2));

			int j = ix1.length - 1;
			ix1[j]++;
			if (m2.getDim(j) != 1)
				ix2[j]++;
			while (ix1[j] == m1.getDim(j)) {
				ix1[j] = 0;
				ix2[j] = 0;
				j--;
				if (j < 0)
					break;
				ix1[j]++;
				if (m2.getDim(j) != 1)
					ix2[j]++;
			}
		}
	}
}
