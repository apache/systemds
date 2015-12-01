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


package org.apache.sysml.runtime.matrix.data;


public class BinaryCellToRowBlockConverter implements Converter<MatrixIndexes, MatrixCell, MatrixIndexes, MatrixBlock>
{
	
	private MatrixIndexes returnIndexes=new MatrixIndexes();
	private MatrixBlock rowBlock = new MatrixBlock();
	private Pair<MatrixIndexes, MatrixBlock> pair=new Pair<MatrixIndexes, MatrixBlock>(returnIndexes, rowBlock);
	private boolean hasValue=false;

	@Override
	public void convert(MatrixIndexes k1, MatrixCell v1) 
	{
		returnIndexes.setIndexes(k1);
		rowBlock.reset(1, 1);
		rowBlock.quickSetValue(0, 0, v1.getValue());
		hasValue=true;
	}

	@Override
	public boolean hasNext() {
		return hasValue;
	}

	@Override
	public Pair<MatrixIndexes, MatrixBlock> next() {
		if(!hasValue)
			return null;
		
		hasValue=false;
		return pair;
	}
	
	@Override
	public void setBlockSize(int rl, int cl) {
	}
}

