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


package org.apache.sysds.runtime.data;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.util.BitSet;

public abstract class DenseBlockFactory
{
	public static DenseBlock createDenseBlock(int rlen, int clen) {
		return createDenseBlock(new int[]{rlen, clen});
	}

	public static DenseBlock createDenseBlock(int rlen, int clen, boolean dedup) {
		return createDenseBlock(new int[]{rlen, clen}, dedup);
	}
	
	public static DenseBlock createDenseBlock(int[] dims) {
		return createDenseBlock(ValueType.FP64, dims);
	}

	public static DenseBlock createDenseBlock(int[] dims, boolean dedup) {
		return createDenseBlock(ValueType.FP64, dims, dedup);
	}
	
	public static DenseBlock createDenseBlock(ValueType vt, int[] dims) {
		DenseBlock.Type type = (UtilFunctions.prod(dims) < Integer.MAX_VALUE) ?
			DenseBlock.Type.DRB : DenseBlock.Type.LDRB;
		return createDenseBlock(vt, type, dims, false);
	}

	public static DenseBlock createDenseBlock(ValueType vt, int[] dims, boolean dedup) {
		DenseBlock.Type type = (UtilFunctions.prod(dims) < Integer.MAX_VALUE) ?
				DenseBlock.Type.DRB : DenseBlock.Type.LDRB;
		return createDenseBlock(vt, type, dims, dedup);
	}

	public static DenseBlock createDenseBlock(BitSet data, int[] dims) {
		return new DenseBlockBoolBitset(dims, data);
	}

	public static DenseBlock createDenseBlock(boolean[] data, int[] dims) {
		return new DenseBlockBoolArray(dims, data);
	}

	public static DenseBlock createDenseBlock(String[] data, int[] dims) {
		return new DenseBlockString(dims, data);
	}

	public static DenseBlock createDenseBlock(double[] data, int[] dims) {
		return new DenseBlockFP64(dims, data);
	}

	public static DenseBlock createDenseBlock(float[] data, int[] dims) {
		return new DenseBlockFP32(dims, data);
	}

	public static DenseBlock createDenseBlock(long[] data, int[] dims) {
		return new DenseBlockInt64(dims, data);
	}

	public static DenseBlock createDenseBlock(int[] data, int[] dims) {
		return new DenseBlockInt32(dims, data);
	}

	public static DenseBlock createDenseBlock(double[] data, int rlen, int clen) {
		return createDenseBlock(data, new int[]{rlen, clen});
	}
	
	public static DenseBlock createDenseBlock(float[] data, int rlen, int clen) {
		return createDenseBlock(data, new int[]{rlen, clen});
	}
	
	public static DenseBlock createDenseBlock(ValueType vt, DenseBlock.Type type, int[] dims, boolean dedup) {
		if( dedup ) {
			switch( type ) {
				case DRB:
					switch(vt) {
						case FP64: return new DenseBlockFP64DEDUP(dims);
						default:
							throw new DMLRuntimeException("Unsupported dense block value type with deduplication enabled: "+vt.name());
					}
				case LDRB:
					switch(vt) {
						default:
							throw new NotImplementedException();
					}
				default:
					throw new DMLRuntimeException("Unexpected dense block type: "+type.name());
			}
		}
		switch( type ) {
			case DRB:
				switch(vt) {
					case FP32: return new DenseBlockFP32(dims);
					case FP64: return new DenseBlockFP64(dims);
					case INT32: return new DenseBlockInt32(dims);
					case INT64: return new DenseBlockInt64(dims);
					case BITSET: return new DenseBlockBoolBitset(dims);
					case BOOLEAN: return new DenseBlockBoolArray(dims);
					case STRING: return new DenseBlockString(dims);
					default:
						throw new DMLRuntimeException("Unsupported dense block value type: "+vt.name());
				}
			case LDRB:
				switch(vt) {
					case FP32: return new DenseBlockLFP32(dims);
					case FP64: return new DenseBlockLFP64(dims);
					case BITSET: return new DenseBlockLBoolBitset(dims);
					case INT32: return new DenseBlockLInt32(dims);
					case INT64: return new DenseBlockLInt64(dims);
					case STRING: return new DenseBlockLString(dims);
					default:
						throw new NotImplementedException();
				}
			default:
				throw new DMLRuntimeException("Unexpected dense block type: "+type.name());
		}
	}

	public static boolean isDenseBlockType(DenseBlock sblock, DenseBlock.Type type) {
		return (getDenseBlockType(sblock) == type);
	}

	public static DenseBlock.Type getDenseBlockType(DenseBlock dblock) {
		return (dblock instanceof DenseBlockDRB) ? DenseBlock.Type.DRB :
			(dblock instanceof DenseBlockLDRB) ? DenseBlock.Type.LDRB : null;
	}

	public static double estimateSizeDenseInMemory(long nrows, long ncols) {
		// estimating the size of a dense matrix by the basic block estimate
		// is a good approximation as for large, partitioned dense blocks, the
		// array headers are in the noise and can be ignored
		return DenseBlockFP64.estimateMemory(nrows, ncols);
	}
}
