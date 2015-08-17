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


package com.ibm.bi.dml.runtime.matrix;

import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;

public class MatrixFormatMetaData extends MatrixDimensionsMetaData 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private InputInfo iinfo;
	private OutputInfo oinfo;
	
	public MatrixFormatMetaData(MatrixCharacteristics mc, OutputInfo oinfo_, InputInfo iinfo_ ) {
		super(mc);
		oinfo = oinfo_;
		iinfo = iinfo_;
	}
	
	public InputInfo getInputInfo() {
		return iinfo;
	}
	
	public OutputInfo getOutputInfo() {
		return oinfo;
	}
	
	@Override
	public Object clone()
	{
		MatrixCharacteristics mc = new MatrixCharacteristics(matchar);
		MatrixFormatMetaData meta = new MatrixFormatMetaData(mc, oinfo, iinfo);
		
		return meta;
	}
}
