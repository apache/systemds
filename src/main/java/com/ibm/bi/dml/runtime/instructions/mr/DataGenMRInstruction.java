/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public abstract class DataGenMRInstruction extends MRInstruction 
{
	
	protected DataGenMethod method;
	protected byte input;
	protected long rows;
	protected long cols;
	protected int rowsInBlock;
	protected int colsInBlock;
	protected String baseDir;
	
	public DataGenMRInstruction(Operator op, DataGenMethod mthd, byte in, byte out, long r, long c, int rpb, int cpb, String dir)
	{
		super(op, out);
		method = mthd;
		input=in;
		rows = r;
		cols = c;
		rowsInBlock = rpb;
		colsInBlock = cpb;
		baseDir = dir;
	}
	
	public DataGenMethod getDataGenMethod() {
		return method;
	}
	
	public byte getInput() {
		return input;
	}

	public long getRows() {
		return rows;
	}

	public long getCols() {
		return cols;
	}

	public int getRowsInBlock() {
		return rowsInBlock;
	}

	public int getColsInBlock() {
		return colsInBlock;
	}

	public String getBaseDir() {
		return baseDir;
	}

	@Override
	public byte[] getInputIndexes() {
		return new byte[]{input};
	}

	@Override
	public byte[] getAllIndexes() {
		return new byte[]{input, output};
	}
}
