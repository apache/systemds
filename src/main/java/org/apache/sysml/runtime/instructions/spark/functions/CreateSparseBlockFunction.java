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
package org.apache.sysml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;

/**
 * General purpose copy function for binary block values. This function can be used in
 * mapValues (copy matrix blocks) to change the internal sparse block representation. 
 * See CopyBlockFunction if no change of SparseBlock.Type required.
 * 
 */
public class CreateSparseBlockFunction implements Function<MatrixBlock,MatrixBlock> 
{
	private static final long serialVersionUID = -4503367283351708178L;
	
	private SparseBlock.Type _stype = null;
	
	public CreateSparseBlockFunction( SparseBlock.Type stype ) {
		_stype = stype;
	}

	@Override
	public MatrixBlock call(MatrixBlock arg0)
		throws Exception 
	{
		//convert given block to CSR representation if in sparse format
		//but allow shallow pass-through if already in CSR representation. 
		if( arg0.isInSparseFormat() && !(arg0 instanceof CompressedMatrixBlock) )
			return new MatrixBlock(arg0, _stype, false);
		else //pass through dense
			return arg0;	
	}
}