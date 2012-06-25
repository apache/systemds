package com.ibm.bi.dml.packagesupport;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.nimble.io.utils.FixedWidthDataset;
import org.nimble.utils.IJVtoCSV;
import org.nimble.utils.I_RowtoIJV;

import com.ibm.bi.dml.packagesupport.Matrix.ValueType;
import com.ibm.bi.dml.sgd.TestDist;


/**
 * Wrapper function for the stochastic gradient descent matrix factorization algorithm. 
 * 
 *
 */
public class SGDWrapper extends PackageFunction {

 
	private static final long serialVersionUID = 4473908199924648673L;
	
	Matrix w;
	Matrix th;
	
	@Override
	public int getNumFunctionOutputs() {
		return 2; 
	}

	@Override
	public FIO getFunctionOutput(int pos) {
		
		if(pos ==0)
			return w;
		if(pos ==1)
			return th;
		
		throw new PackageRuntimeException("invalid output");
	}

	@Override
	public void execute() {
		 
		
		Matrix V = (Matrix) this.getFunctionInput(0);
		String n = (((Scalar)this.getFunctionInput(1)).getValue());
		String m = ((Scalar)this.getFunctionInput(2)).getValue();
		String e = ((Scalar)this.getFunctionInput(3)).getValue();
		String factors = ((Scalar)this.getFunctionInput(4)).getValue();
		
		
		//convert ijv to csv file
		IJVtoCSV dataConv = new IJVtoCSV(10);
		FixedWidthDataset d = new FixedWidthDataset();
		d.setFilePath(V.getFilePath());
		d.setNumFields(1);
		d.setFieldType(0, "java.lang.String");
		FixedWidthDataset outDataset;
		try{
			dataConv.setNumInputDatasets(1);
			dataConv.addInputDataset(d, 0);
			this.getDAGQueue().pushTask(dataConv);
			dataConv = (IJVtoCSV) this.getDAGQueue().waitOnTask(dataConv);
			outDataset = (FixedWidthDataset) dataConv.getOutputDataset(0);
		}
		catch(Exception e1)
		{
			e1.printStackTrace();
			throw new PackageRuntimeException("Eror converting to CSV");
		}

		//invoke sgd code
		String[] args = {n,m,e,factors,outDataset.getFilePath()};
		
		 
		
		try {
			ToolRunner.run(new Configuration(), new TestDist(),args);
		} catch (Exception e1) {
			 
			e1.printStackTrace();
			throw new PackageRuntimeException("Error executing sgd");
		}
		
		//convert output to matrices
		
		d = new FixedWidthDataset();
		d.setFilePath("w.txt");
		d.setNumFields(1);
		d.setFieldType(0, "java.lang.String");
		FixedWidthDataset wDataset;
		try{
			I_RowtoIJV dataConv2 = new I_RowtoIJV(10);
			dataConv2.setNumInputDatasets(1);
			dataConv2.addInputDataset(d, 0);
			this.getDAGQueue().pushTask(dataConv2);
			dataConv2 = (I_RowtoIJV) this.getDAGQueue().waitOnTask(dataConv2);
			wDataset = (FixedWidthDataset) dataConv2.getOutputDataset(0);
		}
		catch(Exception e1)
		{
			e1.printStackTrace();
			throw new PackageRuntimeException("Eror converting to CSV");
		}
		
		d = new FixedWidthDataset();
		d.setFilePath("h.txt");
		d.setNumFields(1);
		d.setFieldType(0, "java.lang.String");
		FixedWidthDataset hDataset;
		try{
			I_RowtoIJV dataConv2 = new I_RowtoIJV(10);
			dataConv2.setNumInputDatasets(1);
			dataConv2.addInputDataset(d, 0);
			this.getDAGQueue().pushTask(dataConv2);
			dataConv2 = (I_RowtoIJV) this.getDAGQueue().waitOnTask(dataConv2);
			hDataset = (FixedWidthDataset) dataConv2.getOutputDataset(0);
		}
		catch(Exception e1)
		{
			e1.printStackTrace();
			throw new PackageRuntimeException("Eror converting to CSV");
		}


		w = new Matrix(wDataset.getFilePath(), V.getNumRows(), Integer.parseInt(factors), ValueType.Double );
		th = new Matrix(hDataset.getFilePath(), V.getNumCols(), Integer.parseInt(factors), ValueType.Double);
		
		
		
		
		
	}

}
