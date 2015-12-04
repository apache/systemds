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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

@SuppressWarnings("rawtypes")
public class AddDummyWeightConverter implements Converter<Writable, Writable, MatrixIndexes, WeightedPair>
{
	
	private Converter toCellConverter=null;
	private WeightedPair outValue=new WeightedPair();
	private Pair<MatrixIndexes, WeightedPair> pair=new Pair<MatrixIndexes, WeightedPair>();
	private int rlen;
	private int clen;
	public AddDummyWeightConverter()
	{
		outValue.setWeight(1.0);
		outValue.setOtherValue(0);
		pair.setValue(outValue);
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public void convert(Writable k1, Writable v1) {
		if(toCellConverter==null)
		{
			if(v1 instanceof Text)
				toCellConverter=new TextToBinaryCellConverter();
			else if(v1 instanceof MatrixBlock)
				toCellConverter=new BinaryBlockToBinaryCellConverter();
			else
				toCellConverter=new IdenticalConverter();
			toCellConverter.setBlockSize(rlen, clen);
		}
		toCellConverter.convert(k1, v1);
	}

	@Override
	public boolean hasNext() {
		return toCellConverter.hasNext();
	}

	@Override
	@SuppressWarnings("unchecked")
	public Pair<MatrixIndexes, WeightedPair> next() {
		Pair<MatrixIndexes, MatrixCell> temp=toCellConverter.next();
		pair.setKey(temp.getKey());
		outValue.setValue(temp.getValue().getValue());
		return pair;
	}

	@Override
	public void setBlockSize(int rl, int cl) {
		
		if(toCellConverter==null)
		{
			rlen=rl;
			clen=cl;
		}else
			toCellConverter.setBlockSize(rl, cl);
	}

}
