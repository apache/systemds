package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;

import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ParamType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.DataFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestVariable;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * Cost Estimator for runtime programs. It uses a offline created performance profile
 * (see PerfTestTool) in order to estimate execution time, memory consumption etc of
 * instructions and program blocks with regard to given data characteristics (e.g., 
 * dimensionality, data format, sparsity) and program parameters (e.g., degree of parallelism).
 * If no performance profile cost function exists for a given TestVariables, TestMeasures, and
 * instructions combination, default values are used. Furthermore, the cost estimator provides
 * basic functionalities for estimation of cardinality and sparsity of intermediate results.
 * 
 * 
 * TODO: complexity corrections for sparse matrices
 */
public class CostEstimator 
{	
	//default parameters
	public static final double DEFAULT_EST_PARALLELISM = 1.0; //default degree of parallelism: serial
	public static final int    FACTOR_NUM_ITERATIONS   = 10; //default problem size
	public static final double DEFAULT_TIME_ESTIMATE   = 5;  //default execution time: 5ms
	public static final double DEFAULT_MEM_ESTIMATE    = 1024; //default memory consumption: 1KB 
	
	//internal estimation parameters
	public static final boolean COMBINE_ESTIMATION_PATHS = false;
	
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
	public static double getEstimate( TestMeasure measure, String instName, double datasize, double sparsity, DataFormat dataformat ) 
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
	public static double getEstimate( TestMeasure measure, String instName, double datasize, double sparsity, double parallelism, DataFormat dataformat ) 
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
	public static double getEstimate( TestMeasure measure, String instName, double dim1, double dim2, double dim3, double sparsity, DataFormat dataformat ) 
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
	public static double getEstimate( TestMeasure measure, String instName, double dim1, double dim2, double dim3, double sparsity, double parallelism, DataFormat dataformat )
		throws DMLRuntimeException
	{
		double ret = -1;
		double datasize = -1;
		
		if( instName.equals("CP"+Lops.OPERAND_DELIMITOR+"ba+*") )
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
			
			if( instName.equals("CP"+Lops.OPERAND_DELIMITOR+"ba+*") )
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
	 * @param measure
	 * @param node
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static double getEstimate( TestMeasure measure, OptNode node ) 
		throws DMLRuntimeException
	{
		double val = -1;
		
		if( node.isLeaf() )
		{
			String str = node.getInstructionName();//node.getParam(ParamType.OPSTRING);
			OptNodeStatistics stats = node.getStatistics();
			DataFormat df = stats.getDataFormat();

			double dim1 = stats.getDim1();
			double dim2 = Math.max(stats.getDim2(), stats.getDim3()); //using max useful if just one side known
			double dim3 = stats.getDim4();
			double sparsity = stats.getSparsity();
			val = getEstimate(measure, str, dim1, dim2, dim3, sparsity, df);
			
			//FIXME just for test until cost functions for MR are trained
			if( node.getExecType() == ExecType.MR )
				val = 60000; //1min or 60k
			
			//System.out.println("measure="+measure+", operation="+str+", val="+val);
		}
		else
		{
			//aggreagtion methods for different program block types and measure types
			String tmp = null;
			double N = -1;
			switch ( measure )
			{
				case EXEC_TIME:
					switch( node.getNodeType() )
					{
						case GENERIC:
							val = getSumEstimate(measure, node.getChilds()); 
							break;
						case IF:
							val = getMaxEstimate(measure, node.getChilds()); 
							break;
						case WHILE:
							val = FACTOR_NUM_ITERATIONS * getSumEstimate(measure, node.getChilds()); 
							break;
						case FOR:
							tmp = node.getParam(ParamType.NUM_ITERATIONS);
							N = (tmp!=null) ? (double)Integer.parseInt(tmp) : FACTOR_NUM_ITERATIONS; 
							val = N * getSumEstimate(measure, node.getChilds());
							break; 
						case PARFOR:
							tmp = node.getParam(ParamType.NUM_ITERATIONS);
							N = (tmp!=null) ? (double)Integer.parseInt(tmp) : FACTOR_NUM_ITERATIONS; 
							val = N * getSumEstimate(measure, node.getChilds()) / node.getK(); 
							break;						
					}
					break;
					
				case MEMORY_USAGE:
					switch( node.getNodeType() )
					{
						case GENERIC:
						case IF:
						case WHILE:
						case FOR:
							val = getMaxEstimate(measure, node.getChilds()); 
							break;
						case PARFOR:
							if( node.getExecType() == ExecType.MR )
								val = getMaxEstimate(measure, node.getChilds()); //executed in different JVMs
							else if ( node.getExecType() == ExecType.CP )
								val = getMaxEstimate(measure, node.getChilds()) * node.getK(); //everything executed within 1 JVM
							break;
					}
					break;
			}
		}
		
		return val;
	}

	/**
	 * 
	 * @param measure
	 * @param nodes
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static double getMaxEstimate( TestMeasure measure, ArrayList<OptNode> nodes ) 
		throws DMLRuntimeException
	{
		double max = Double.MIN_VALUE;
		for( OptNode n : nodes )
		{
			double tmp = getEstimate( measure, n );
			if( tmp > max )
				max = tmp;
		}
		return max;
	}
	
	/**
	 * 
	 * @param measure
	 * @param nodes
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static double getSumEstimate( TestMeasure measure, ArrayList<OptNode> nodes ) 
		throws DMLRuntimeException
	{
		double sum = 0;
		for( OptNode n : nodes )
			sum += getEstimate( measure, n );
		return sum;	
	}

	/**
	 * 
	 * @param measure
	 * @return
	 */
	private static double getDefaultEstimate(TestMeasure measure) 
	{
		double val = -1;
		
		switch( measure )
		{
			case EXEC_TIME: val = DEFAULT_TIME_ESTIMATE; break;
			case MEMORY_USAGE: val = DEFAULT_MEM_ESTIMATE; break;
		}
		
		return val;
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
		
		//System.out.println("f(Dx="+x1+") = "+val11);
		//System.out.println("f(Dd="+d1+") = "+val12);
		//System.out.println("f(Sx="+x2+") = "+val21);
		//System.out.println("f(Sd="+d2+") = "+val22);
		
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

	/**
	 * 
	 * @param plan
	 * @param n
	 * @return
	 */
	public static double computeLocalParBound(OptTree plan, OptNode n) 
	{
		return Math.floor(rComputeLocalValueBound(plan.getRoot(), n, plan.getCK()));		
	}

	/**
	 * 
	 * @param plan
	 * @param n
	 * @return
	 */
	public static double computeLocalMemoryBound(OptTree plan, OptNode n) 
	{
		return rComputeLocalValueBound(plan.getRoot(), n, plan.getCM());
	}

	/**
	 * 
	 * @param pn
	 * @return
	 */
	public static double getMinMemoryUsage(OptNode pn) 
	{
		// TODO implement for DP enum optimizer
		throw new RuntimeException("Not implemented yet.");
	}

	/**
	 * 
	 * @param current
	 * @param node
	 * @param currentVal
	 * @return
	 */
	private static double rComputeLocalValueBound( OptNode current, OptNode node, double currentVal )
	{
		if( current == node ) //found node
			return currentVal;
		else if( current.isLeaf() ) //node not here
			return -1; 
		else
		{
			switch( current.getNodeType() )
			{
				case GENERIC:
				case IF:
				case WHILE:
				case FOR:
					for( OptNode c : current.getChilds() ) 
					{
						double lval = rComputeLocalValueBound(c, node, currentVal);
						if( lval > 0 )
							return lval;
					}
					break;
				case PARFOR:
					for( OptNode c : current.getChilds() ) 
					{
						double lval = rComputeLocalValueBound(c, node, currentVal/current.getK());
						if( lval > 0 )
							return lval;
					}
					break;
			}
		}
			
		return -1;
	}
	
}
