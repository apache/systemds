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

package org.apache.sysml.runtime.instructions.mr;

import java.util.ArrayList;

import org.apache.sysml.lops.PartialAggregate.CorrectionLocationType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.DistributedCacheInput;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class GroupedAggregateMInstruction extends BinaryMRInstructionBase implements IDistributedCacheConsumer
{	
	private int _ngroups = -1;
	
	public GroupedAggregateMInstruction(Operator op, byte in1, byte in2, byte out, int ngroups, String istr)
	{
		super(op, in1, in2, out);
		_ngroups = ngroups;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static GroupedAggregateMInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionParts ( str );
		InstructionUtils.checkNumFields(parts, 5);
		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		byte out = Byte.parseByte(parts[3]);
		int ngroups = Integer.parseInt(parts[4]);
		//partitioning ignored
		
		Operator op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
		
		return new GroupedAggregateMInstruction(op, in1, in2, out, ngroups, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input1);
		if( blkList == null ) 
			return;
		
		for(IndexedMatrixValue in1 : blkList)
		{
			if(in1 == null)
				continue;
		
			DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(input2);
			
			//get all inputs
			MatrixIndexes ix = in1.getIndexes();
			MatrixBlock groups = (MatrixBlock)dcInput.getDataBlock((int)ix.getRowIndex(), 1).getValue();
			
			//output blocked result
			int brlen = dcInput.getNumRowsPerBlock();
			int bclen = dcInput.getNumColsPerBlock();
			
			//execute map grouped aggregate operations
			ArrayList<IndexedMatrixValue> outlist = new ArrayList<IndexedMatrixValue>();
			OperationsOnMatrixValues.performMapGroupedAggregate(getOperator(), in1, groups, _ngroups, brlen, bclen, outlist);
			
			//output all result blocks
			for( IndexedMatrixValue out : outlist ) {
				cachedValues.add(output, out);
			}			
		}	
	}
	

	@Override //IDistributedCacheConsumer
	public boolean isDistCacheOnlyIndex( String inst, byte index )
	{
		return (index==input2 && index!=input1);
	}
	
	@Override //IDistributedCacheConsumer
	public void addDistCacheIndex( String inst, ArrayList<Byte> indexes )
	{
		indexes.add(input2);
	}
	
	public void computeOutputCharacteristics(MatrixCharacteristics mcIn, MatrixCharacteristics mcOut) {
		mcOut.set(_ngroups, mcIn.getCols(), mcIn.getRowsPerBlock(), mcIn.getColsPerBlock());
	}
}
