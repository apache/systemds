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

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;


public class BinaryCellToTextConverter 
implements Converter<MatrixIndexes, MatrixCell, NullWritable, Text>
{
	
	private Text value=new Text();
	private Pair<NullWritable, Text> pair=new Pair<NullWritable, Text>(NullWritable.get(), value);
	private boolean hasValue=false;

	@Override
	public void convert(MatrixIndexes k1, MatrixCell v1) {
		double v=((MatrixCell)v1).getValue();
		value.set(k1.getRowIndex()+" "+k1.getColumnIndex()+" "+v);
		hasValue=true;
	}

	@Override
	public boolean hasNext() {
		return hasValue;
	}

	@Override
	public Pair<NullWritable, Text> next() {
		if(!hasValue)
			return null;
		
		hasValue=false;
		return pair;
	}

	@Override
	public void setBlockSize(int rl, int cl) {
	}
}
