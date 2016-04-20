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


import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;



public class FrameBinaryBlockToTextCellConverter implements 
Converter<LongWritable, FrameBlock, NullWritable, Text>
{	
	private FrameBlock _block;
	private int nextRow=-1;
	private int nextCol=-1;
	private LongWritable startIndexes=new LongWritable();
	private boolean hasValue=false;
	
	private Text value=new Text();
	private Pair<NullWritable, Text> pair=new Pair<NullWritable, Text>(NullWritable.get(), value);
	
	private void reset()
	{
		nextRow=-1;
		nextCol=-1;
	}
	
	/**
	 * Before calling convert, please make sure to setBlockSize(brlen, bclen);
	 */
	@Override
	public void convert(LongWritable k1, FrameBlock v1) {
		reset();
		startIndexes.set(k1.get()); 
		nextRow=0;
		nextCol=0;
		hasValue = true;
		_block = v1;
		hasValue=(v1.getNumRows()>0);
	}

	@Override
	public boolean hasNext() {

		hasValue=false;

		if(_block.getNumRows() > 0 && nextRow<_block.getNumRows()) {
			boolean bNextCell=true;
			while(_block.get(nextRow, nextCol) == null && bNextCell)
				bNextCell=incrCellIndex();
	
			if(_block.get(nextRow, nextCol) != null && bNextCell)
				hasValue = true;
		}
		
		return hasValue;
	}

	@Override
	public Pair<NullWritable, Text> next() {
		if(!hasValue || _block.getNumRows() == 0)
			return null;
		long i, j;

		i=startIndexes.get() + nextRow;
		j=nextCol+1;
		Object obj = _block.get(nextRow, nextCol);
		if(obj != null)
			value.set(i+" "+j+" "+obj.toString());

		incrCellIndex();

		return pair;
	}

	public void setBlockSize(int nr, int nc) {
	}
	
	private boolean  incrCellIndex() {
		boolean bNextCellExists = true;
		if(nextCol == (_block.getNumColumns()-1)) {
			nextCol = 0;
			++nextRow;
			if(nextRow == (_block.getNumRows()))
				bNextCellExists = false;
		}
		else
			++nextCol;
		return bNextCellExists;
	}
}
