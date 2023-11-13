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
package org.apache.sysds.runtime.compress.lib;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBin;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;

public class CLALibBinCompress {
	public static ColumnEncoderBin.BinMethod binMethod = ColumnEncoderBin.BinMethod.EQUI_WIDTH;
	public static Pair<MatrixBlock, FrameBlock> binCompress(CacheBlock<?> X, MatrixBlock d, int k){
		// Create transform spec acc to binMethod
		String spec = createSpec(d);

		// Apply compressed transform encode using spec
		MultiColumnEncoder encoder = //
			EncoderFactory.createEncoder(spec, null, X.getNumColumns(), null);
		MatrixBlock binned = encoder.encode(X, k, true);

		// Get metadata from transformencode
		FrameBlock meta = new FrameBlock(X.getNumColumns(), Types.ValueType.STRING);
		encoder.initMetaData(meta);
		FrameBlock newMeta = encoder.getMetaData(meta, k);

		// FIXME Optional: recompress, else can be removed (lines 54-55) once fixed compression
		if(X instanceof MatrixBlock) {
			Pair<MatrixBlock, CompressionStatistics> recompressed = CompressedMatrixBlockFactory.compress(binned, k);
			return new ImmutablePair<>(recompressed.getKey(), newMeta);
		}
		else
			return new ImmutablePair<>(binned, newMeta);
	}

	private static String createSpec(MatrixBlock d) {
		d.sparseToDense();
		double[] values = d.getDenseBlockValues();

		String binning = binMethod.toString();

		StringBuilder stringBuilder = new StringBuilder();
		stringBuilder.append("{\"ids\":true,\"bin\":[");
		for(int i = 0; i < values.length; i++) {
			stringBuilder.append(String.format("{\"id\":%d,\"method\":\"%s\",\"numbins\":%d}", i + 1, binning, (int)values[i]));
			if(i + 1 < values.length)
				stringBuilder.append(',');
		}
		stringBuilder.append("]}");
		return stringBuilder.toString();
	}
}
