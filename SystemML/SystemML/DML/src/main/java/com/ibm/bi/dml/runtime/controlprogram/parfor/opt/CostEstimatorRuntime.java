/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;


import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.DataFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestVariable;

/**
 * Cost Estimator for runtime programs. It uses a offline created performance profile
 * (see PerfTestTool) in order to estimate execution time, memory consumption etc of
 * instructions and program blocks with regard to given data characteristics (e.g., 
 * dimensionality, data format, sparsity) and program parameters (e.g., degree of parallelism).
 * If no performance profile cost function exists for a given TestVariables, TestMeasures, and
 * instructions combination, default values are used. Furthermore, the cost estimator provides
 * basic functionalities for estimation of cardinality and sparsity of intermediate results.
 * 
 * TODO: inst names as constants in perftesttool
 * TODO: complexity corrections for sparse matrices
 */
public class CostEstimatorRuntime extends CostEstimator
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//internal estimation parameters
	public static final boolean COMBINE_ESTIMATION_PATHS = false;
		
	@Override
	public double getLeafNodeEstimate( TestMeasure measure, OptNode node ) 
		throws DMLRuntimeException
	{
		double val = -1;

		String str = node.getInstructionName();//node.getParam(ParamType.OPSTRING);
		OptNodeStatistics stats = node.getStatistics();
		DataFormat df = stats.getDataFormat();

		double dim1 = stats.getDim1();
		double dim2 = Math.max(stats.getDim2(), stats.getDim3()); //using max useful if just one side known
		double dim3 = stats.getDim4();
		double sparsity = stats.getSparsity();
		val = getEstimate(measure, str, dim1, dim2, dim3, sparsity, df);
		
		//FIXME just for test until cost functions for MR are trained
		if( node.getExecType() == OptNode.ExecType.MR )
			val = 60000; //1min or 60k
		
		//System.out.println("measure="+measure+", operation="+str+", val="+val);
		
		return val;
	}
	
	@Override
	public double getLeafNodeEstimate( TestMeasure measure, OptNode node, ExecType et ) 
			throws DMLRuntimeException
	{
		//TODO for the moment invariant of et
		
		return getLeafNodeEstimate(measure, node);
	}
	
	/**
	 * 
	 * @param measure
	 * @param instName
	 * @param datasize
	 * @param sparsity
	 * @param dataformat
	 * @return
	 * @throws DMLRuntimeException
	 */
	public double getEstimate( TestMeasure measure, String instName, double datasize, double sparsity, DataFormat dataformat ) 
		throws DMLRuntimeException
	{
		return getEstimate(measure, instName, datasize, sparsity, DEFAULT_EST_PARALLELISM, dataformat);
	}
	
	/**
	 * 
	 * @param measure
	 * @param instName
	 * @param datasize
	 * @param sparsity
	 * @param parallelism
	 * @param dataformat
	 * @return
	 * @throws DMLRuntimeException
	 */
	public double getEstimate( TestMeasure measure, String instName, double datasize, double sparsity, double parallelism, DataFormat dataformat ) 
		throws DMLRuntimeException
	{
		double dim = Math.sqrt( datasize );		
		return getEstimate(measure, instName, dim, dim, dim, sparsity, parallelism, dataformat);
	}
	
	/**
	 * 
	 * @param measure
	 * @param instName
	 * @param dim1
	 * @param dim2
	 * @param dim3
	 * @param sparsity
	 * @param dataformat
	 * @return
	 * @throws DMLRuntimeException
	 */
	public double getEstimate( TestMeasure measure, String instName, double dim1, double dim2, double dim3, double sparsity, DataFormat dataformat ) 
		throws DMLRuntimeException
	{
		return getEstimate(measure, instName, dim1, dim2, dim3, sparsity, DEFAULT_EST_PARALLELISM, dataformat);
	}
	
	/**
	 * 
	 * @param measure
	 * @param instName
	 * @param dim1
	 * @param dim2
	 * @param dim3
	 * @param sparsity
	 * @param parallelism
	 * @param dataformat
	 * @return
	 * @throws DMLRuntimeException
	 */
	public double getEstimate( TestMeasure measure, String instName, double dim1, double dim2, double dim3, double sparsity, double parallelism, DataFormat dataformat )
		throws DMLRuntimeException
	{
		double ret = -1;
		double datasize = -1;
		
		if( instName.equals("CP"+Lop.OPERAND_DELIMITOR+"ba+*") )
			datasize = (dim1*dim2 + dim2*dim3 + dim1*dim3)/3;
		else
			datasize = dim1*dim2;
		
		//get basic cost functions
		CostFunction df = PerfTestTool.getCostFunction(instName, measure, TestVariable.DATA_SIZE, dataformat);
		CostFunction sf = PerfTestTool.getCostFunction(instName, measure, TestVariable.SPARSITY, dataformat);
		
		if( df == null || sf == null )
		{
			return getDefaultEstimate( measure );
		}
		
		//core merge datasize and sparsity
		
		//complexity corrections (inc. offset correction)
		if( !df.isMultiDim() ) 
		{
			
			ret =  aggregate( df, sf, 
			          datasize, PerfTestTool.DEFAULT_DATASIZE, 
			          sparsity, PerfTestTool.DEFAULT_SPARSITY );	
			
			//System.out.println("before correction = "+ret);
			
			double offset = df.estimate(0);
			double ddim   = Math.sqrt(datasize);
			double assumedC = -1;
			double realC = -1;
			
			if( instName.equals("CP"+Lop.OPERAND_DELIMITOR+"ba+*") )
			{
				switch( measure )
				{
					case EXEC_TIME:
						assumedC = 2*ddim * ddim * ddim + ddim * ddim;
						if( dataformat==DataFormat.DENSE )
							realC = 2*dim1 * dim2 * dim3 + dim1 * dim3;
						else if( dataformat==DataFormat.SPARSE ) 
							realC = 2*dim1 * dim2 * dim3 + dim1 * dim3;
						break;
					case MEMORY_USAGE:
						assumedC = 3*ddim*ddim;
						if( dataformat==DataFormat.DENSE )
							realC = dim1 * dim2 + dim2 * dim3 + dim1 * dim3;
					    else if( dataformat==DataFormat.SPARSE ) 
					    	realC = dim1 * dim2 + dim2 * dim3 + dim1 * dim3;
						break;
				}
				//actual correction (without offset)
				ret = (ret-offset) * realC/assumedC + offset;
			}
			
			/*NEW COMPLEXITY CORRECTIONS GO HERE*/
		}
		else
		{
			double ddim = Math.sqrt(PerfTestTool.DEFAULT_DATASIZE);
			
			ret =  aggregate( df, sf, 
			          new double[]{dim1,dim2,dim3}, new double[]{ddim,ddim,ddim}, 
			          sparsity, PerfTestTool.DEFAULT_SPARSITY );	
			
		}

		return ret;
	}

	/**
	 * 
	 * @param f1
	 * @param f2
	 * @param x1
	 * @param d1
	 * @param x2
	 * @param d2
	 * @return
	 */
	private static double aggregate( CostFunction f1, CostFunction f2, double x1, double d1, double x2, double d2 )
	{
		double val11 = f1.estimate(x1);
		double val12 = f1.estimate(d1);
		double val21 = f2.estimate(x2);
		double val22 = f2.estimate(d2);
		
		//estimate combined measure
		double ret;
		if( COMBINE_ESTIMATION_PATHS )
			ret = ((val11 * val21 / val22) + (val21 * val11 / val12)) / 2;
		else
			ret = (val11 * val21 / val22);
		
		return ret;
	}

	/**
	 * 
	 * @param f1
	 * @param f2
	 * @param x1
	 * @param d1
	 * @param x2
	 * @param d2
	 * @return
	 */
	private static double aggregate( CostFunction f1, CostFunction f2, double[] x1, double[] d1, double x2, double d2 )
	{
		double val11 = f1.estimate(x1);
		double val12 = f1.estimate(d1);
		double val21 = f2.estimate(x2);
		double val22 = f2.estimate(d2);
		
		//estimate combined measure
		double ret;
		if( COMBINE_ESTIMATION_PATHS )
			ret = ((val11 * val21 / val22) + (val21 * val11 / val12)) / 2;
		else
			ret = (val11 * val21 / val22);
		
		return ret;
	}

	
}
