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


package org.apache.sysml.runtime.matrix.data;

public abstract class DenseBlockFactory
{
	public static DenseBlock createDenseBlock(int rlen, int clen) {
		DenseBlock.Type type = ((long)rlen*clen < Integer.MAX_VALUE) ?
			DenseBlock.Type.DRB : DenseBlock.Type.LDRB;
		return createDenseBlock(type, rlen, clen);
	}

	public static DenseBlock createDenseBlock(double[] data, int rlen, int clen) {
		return new DenseBlockDRB(data, rlen, clen);
	}
	
	public static DenseBlock createDenseBlock(DenseBlock.Type type, int rlen, int clen) {
		switch( type ) {
			case DRB: return new DenseBlockDRB(rlen, clen);
			case LDRB: return new DenseBlockLDRB(rlen, clen);
			default:
				throw new RuntimeException("Unexpected dense block type: "+type.name());
		}
	}

	public static boolean isDenseBlockType(DenseBlock sblock, DenseBlock.Type type) {
		return (getDenseBlockType(sblock) == type);
	}

	public static DenseBlock.Type getDenseBlockType(DenseBlock dblock) {
		return (dblock instanceof DenseBlockDRB) ? DenseBlock.Type.DRB :
			(dblock instanceof DenseBlockLDRB) ? DenseBlock.Type.LDRB : null;
	}
}
