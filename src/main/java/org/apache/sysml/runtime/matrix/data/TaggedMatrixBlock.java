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


package com.ibm.bi.dml.runtime.matrix.data;

public class TaggedMatrixBlock extends TaggedMatrixValue
{
		
	public TaggedMatrixBlock(MatrixBlock b, byte t) {
		super(b, t);
	}

	public TaggedMatrixBlock()
	{        
        tag=-1;
     	base=new MatrixBlock();
	}

	public TaggedMatrixBlock(TaggedMatrixBlock that) {
		tag=that.getTag();
		base=new MatrixBlock();
		base.copy(that.getBaseObject());
	}
}