/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.udf.lib;

import java.io.DataOutputStream;

import org.nimble.algorithms.sparsekmeans.SparseKMeansTask;
import org.nimble.hadoop.HDFSFileManager;
import org.nimble.io.utils.FixedWidthDataset;
import org.nimble.io.utils.ObjectDataset;
import org.nimble.algorithms.sparsekmeans.MatrixtoSparseBlockMatrix;
import org.nimble.algorithms.sparsekmeans.SparseHashMatrix;

import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Scalar;
import com.ibm.bi.dml.udf.Matrix.ValueType;



/**
 * Wrapper for kmeans clustering. This class first converts the 
 * matrix into a fixedwidthdataset appropriate for kmeansTask. 
 * It then invokes kMeansTask and return the result as a matrix.
 * It takes 2 parameters, the input matrix, and number of centers k
 * It returns a matrix representing the k centers
 * 
 *
 */
public class kMeansWrapper extends PackageFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	//to be used when aggregating rows.
	private static final long BLOCK_SIZE = 10000000;
	private static final long serialVersionUID = 6799705939733343000L;
	private static final String OUTPUT_FILE = "kMeansWrapperOutput";
	
	Matrix outkcenters; 

	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		return outkcenters;
	}

	@Override
	public void execute() {
		try
		{
			//preprocess matrix to convert to fixed width dataset
			Matrix m = (Matrix) this.getFunctionInput(0);
			
			 
			
			MatrixtoSparseBlockMatrix dataConv = new MatrixtoSparseBlockMatrix(m.getNumRows(), m.getNumCols(),BLOCK_SIZE, BLOCK_SIZE, 52);
			FixedWidthDataset d = new FixedWidthDataset();
			d.setFilePath(m.getFilePath());
			d.setNumFields(1);
			d.setFieldType(0, "java.lang.String");
			dataConv.setNumInputDatasets(1);
			dataConv.addInputDataset(d, 0);
			this.getDAGQueue().pushTask(dataConv);
			dataConv = (MatrixtoSparseBlockMatrix) this.getDAGQueue().waitOnTask(dataConv);
			ObjectDataset kMeansDataset = dataConv.getProcessedDataset();
			
			//execute kMeans clustering on converted dataset
			Scalar k = (Scalar) this.getFunctionInput(1);
			SparseKMeansTask kmeans;
			if(this.getNumFunctionInputs() > 2)
			{
			  SparseHashMatrix mat = new SparseHashMatrix(Integer.parseInt(k.getValue()), (int) m.getNumCols());
			  mat.load(((Matrix)this.getFunctionInput(2)).getMatrixAsDoubleArray());
			  kmeans = new SparseKMeansTask(mat, Integer.parseInt(k.getValue()), (int) m.getNumCols(), 100);
			}
			else
			{
			  kmeans = new SparseKMeansTask(null, Integer.parseInt(k.getValue()), (int) m.getNumCols(), 100);
			}
			kmeans.setWithoutConfig();
			kmeans.setNumInputDatasets(1);
			kmeans.addInputDataset(kMeansDataset, 0);
			this.getDAGQueue().pushTask(kmeans);
			kmeans = (SparseKMeansTask) this.getDAGQueue().waitOnTask(kmeans);
			
			//write out centers
			String fname = createOutputFilePathAndName( OUTPUT_FILE );
			DataOutputStream ostream = HDFSFileManager.getOutputStreamStatic(fname, true);
			for(Integer row: kmeans.prevkCenters.getRows())
			{
				int [] col_indices = kmeans.prevkCenters.getColumnIndices(row);
				double [] vals = kmeans.prevkCenters.getValues(row);
				for(int j=0; j < col_indices.length; j++)
				{
				    ostream.writeBytes((row+1) + " " + (col_indices[j]+1) + " " + vals[j] + "\n");	
				}
			}
			
			ostream.close();
			//setup output to be returned
			outkcenters = new Matrix(fname, Integer.parseInt(k.getValue()) , m.getNumCols(), ValueType.Double);
			

		}
		catch(Exception e)
		{
			throw new PackageRuntimeException("Error executing kMeans clustering",e);
		}


	}

}
