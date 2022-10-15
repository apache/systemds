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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Write block for serializing either a instance of MatrixBlock or CompressedMatrixBlock, To allow spark to read in
 * either or.
 */
public class CompressedWriteBlock implements WritableComparable<CompressedWriteBlock> {

	public MatrixBlock mb;

	private enum CONTENT {
		Comp, MB;
	}

	/**
	 * Write block used to point to a underlying instance of CompressedMatrixBlock or MatrixBlock, Unfortunately spark
	 * require a specific object type to serialize therefore we use this class.
	 */
	public CompressedWriteBlock() {
		// Empty constructor ... used for serialization of the block.
	}

	public CompressedWriteBlock(MatrixBlock mb) {
		this.mb = mb;
	}

	@Override
	public void write(DataOutput out) throws IOException {

		if(mb instanceof CompressedMatrixBlock)
			out.writeByte(CONTENT.Comp.ordinal());
		else
			out.writeByte(CONTENT.MB.ordinal());
		mb.write(out);

	}

	@Override
	public void readFields(DataInput in) throws IOException {
		switch(CONTENT.values()[in.readByte()]) {
			case Comp:
				mb = CompressedMatrixBlock.read(in);
				break;
			case MB:
				mb = new MatrixBlock();
				mb.readFields(in);
				break;
		}
	}

	public MatrixBlock get() {
		return mb;
	}

	@Override
	public int compareTo(CompressedWriteBlock arg0) {
		throw new RuntimeException("CompareTo should never be called for WriteBlock.");
	}

}
