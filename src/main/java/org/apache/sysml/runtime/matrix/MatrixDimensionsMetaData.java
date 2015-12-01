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


package org.apache.sysml.runtime.matrix;

public class MatrixDimensionsMetaData extends MetaData 
{
	protected MatrixCharacteristics matchar;
	
	public MatrixDimensionsMetaData() {
		matchar = null;
	}
	
	public MatrixDimensionsMetaData(MatrixCharacteristics mc) {
		matchar = mc;
	}
	
	public MatrixCharacteristics getMatrixCharacteristics() {
		return matchar;
	}
	
	public void setMatrixCharacteristics(MatrixCharacteristics mc) {
		matchar = mc;
	}
	
	@Override
	public boolean equals (Object anObject)
	{
		if (anObject instanceof MatrixDimensionsMetaData)
		{
			MatrixDimensionsMetaData md = (MatrixDimensionsMetaData) anObject;
			return (matchar.equals (md.matchar));
		}
		else
			return false;
	}
	
	@Override
	public int hashCode()
	{
		//use identity hash code
		return super.hashCode();
	}

	@Override
	public String toString() {
		return "[rows = " + matchar.getRows() + 
			   ", cols = " + matchar.getCols() + 
			   ", rpb = " + matchar.getRowsPerBlock() + 
			   ", cpb = " + matchar.getColsPerBlock() + "]"; 
	}
	
	@Override
	public Object clone()
	{
		MatrixCharacteristics mc = new MatrixCharacteristics(matchar);
		MatrixDimensionsMetaData ret = new MatrixDimensionsMetaData(mc);
		
		return ret;
	}
}
