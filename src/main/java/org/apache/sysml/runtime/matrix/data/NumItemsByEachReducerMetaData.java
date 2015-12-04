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

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;

public class NumItemsByEachReducerMetaData extends MatrixDimensionsMetaData 
{
	
	private long[] numItems=null;
	private int partitionOfZero=-1;
	private long numberOfZero=0;
	
	public NumItemsByEachReducerMetaData(MatrixCharacteristics mc, long[] nums, int part0, long num0)
	{
		super(mc);
		numItems=nums;
		partitionOfZero=part0;
		numberOfZero=num0;
	}
	
	public NumItemsByEachReducerMetaData(MatrixCharacteristics mc, long[] nums) {
		super(mc);
		numItems=nums;
	}
	
	public long[] getNumItemsArray()
	{
		return numItems;
	}
	
	public int getPartitionOfZero()
	{
		return partitionOfZero;
	}
	
	public long getNumberOfZero()
	{
		return numberOfZero;
	}
	
	@Override
	public Object clone()
	{
		MatrixCharacteristics mc = new MatrixCharacteristics(matchar);
		NumItemsByEachReducerMetaData ret = new NumItemsByEachReducerMetaData(mc, numItems, partitionOfZero, numberOfZero);
	
		return ret;
	}
}
