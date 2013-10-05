/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.packagesupport;

import com.ibm.bi.dml.packagesupport.FIO;
import com.ibm.bi.dml.packagesupport.Matrix;
import com.ibm.bi.dml.packagesupport.PackageFunction;
import com.ibm.bi.dml.packagesupport.PackageRuntimeException;
import com.ibm.bi.dml.packagesupport.Matrix.ValueType;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;

public class DynamicProjectMatrixCP extends PackageFunction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;	
	private final String OUTPUT_FILE = "DynProjectMatrixWrapperOutput2D";
	
	private Matrix _ret; 
	
	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;
	}

	@Override
	public FIO getFunctionOutput(int pos) 
	{
		return _ret;
	}

	@Override
	public void execute() 
	{
		try
		{						
			Matrix mD = (Matrix) this.getFunctionInput(0);
			Matrix mC = (Matrix) this.getFunctionInput(1);
			MatrixBlock mbD = mD.getMatrixObject().acquireRead();
			MatrixBlock mbC = mC.getMatrixObject().acquireRead();
			int rows = mbC.getNumColumns();
			int cols = mbC.getNumColumns();
			
			String dir = createOutputFilePathAndName( OUTPUT_FILE );
			
			MatrixBlock mb = null;
			
			if( mbD.getNumColumns()==1 ) //VECTOR
			{
				cols=1;
				mb = new MatrixBlock(rows,cols,false);
				
				for(int i=0; i < rows; i++)
				{
					int ix1 = (int)mbC.quickGetValue(0, i)-1;
					double val = mbD.quickGetValue(ix1, 0);
					mb.quickSetValue(i, 0, val);	
				}	
			}
			else //MATRIX
			{
				mb = new MatrixBlock(rows,cols,false);
				
				for(int i=0; i < rows; i++)
				{
					int ix1 = (int)mbC.quickGetValue(0, i)-1;
					for(int j=0; j < cols; j++)
					{
						int ix2 = (int)mbC.quickGetValue(0, j)-1;
						double val = mbD.quickGetValue(ix1, ix2);
						mb.quickSetValue(i, j, val);	
					}			
				}
			}
			_ret = new Matrix(dir, rows, cols, ValueType.Double);			
			_ret.setMatrixDoubleArray(mb, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
	
			mD.getMatrixObject().release();
			mC.getMatrixObject().release();
		}
		catch(Exception e)
		{
			throw new PackageRuntimeException("Error executing dynamic project of matrix", e);
		}
	}	
}
