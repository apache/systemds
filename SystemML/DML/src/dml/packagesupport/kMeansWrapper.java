package dml.packagesupport;

import java.io.DataOutputStream;

import org.nimble.algorithms.kmeans.SparseKMeansTask;
import org.nimble.hadoop.HDFSFileManager;
import org.nimble.io.utils.FixedWidthDataset;
import org.nimble.io.utils.ObjectDataset;
import org.nimble.utils.MatrixtoSparseBlockMatrix;
import org.nimble.utils.SparseHashMatrix;

import dml.packagesupport.Matrix.ValueType;


/**
 * Wrapper for kmeans clustering. This class first converts the 
 * matrix into a fixedwidthdataset appropriate for kmeansTask. 
 * It then invokes kMeansTask and return the result as a matrix.
 * It takes 2 parameters, the input matrix, and number of centers k
 * It returns a matrix representing the k centers
 * @author aghoting
 *
 */
public class kMeansWrapper extends PackageFunction {


	//to be used when aggregating rows.
	final long BLOCK_SIZE = 10000000;
	private static final long serialVersionUID = 6799705939733343000L;
	final String OUTPUT_FILE = "PackageSupport/kMeansWrapperOutput";
	
	Matrix outkcenters; 

	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FIO getFunctionOutput(int pos) {
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
			DataOutputStream ostream = HDFSFileManager.getOutputStreamStatic(OUTPUT_FILE, true);
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
			outkcenters = new Matrix(OUTPUT_FILE, Integer.parseInt(k.getValue()) , m.getNumCols(), ValueType.Double);
			

		}
		catch(Exception e)
		{
			e.printStackTrace();
			throw new PackageRuntimeException("Error executing kMeans clustering");
		}


	}

}
