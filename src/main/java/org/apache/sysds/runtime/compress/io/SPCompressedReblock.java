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

package org.apache.sysds.runtime.compress.io;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.DataCharacteristics;

public class SPCompressedReblock {

	@SuppressWarnings({"unchecked"})
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> getRDDHandle(SparkExecutionContext sec, String inID,
		String outID) {

		MatrixObject mo = sec.getMatrixObject(inID);
		DataCharacteristics mc = sec.getDataCharacteristics(inID);
		DataCharacteristics mcOut = sec.getDataCharacteristics(outID);
		// BINARY BLOCK <- BINARY BLOCK (different sizes)
		JavaPairRDD<MatrixIndexes, MatrixBlock> in1 = (JavaPairRDD<MatrixIndexes, MatrixBlock>) sec
			.getRDDHandleForMatrixObject(mo, FileFormat.COMPRESSED);
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDConverterUtils.binaryBlockToBinaryBlock(in1, mc, mcOut);
		return out;

	}
}
