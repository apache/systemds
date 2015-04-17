/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.context;

import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.MLContext;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.spark.fix.CopyBlockFunction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;


public class SparkExecutionContext extends ExecutionContext
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	//executor memory and relative fractions as obtained from the spark configuration
	private static long _memExecutors = -1;
	private static double _memRatioData = -1;
	private static double _memRatioShuffle = -1;
	
	// TODO: This needs to be debugged further. For now getting around the problem with Singleton
	// Only one SparkContext may be active per JVM. You must stop() the active SparkContext before creating a new one. 
	// This limitation may eventually be removed; see SPARK-2243 for more details.
	private static JavaSparkContext _singletonSpctx = null;
	
	private JavaSparkContext _spctx = null; 
	
	protected SparkExecutionContext(Program prog) 
	{
		//protected constructor to force use of ExecutionContextFactory
		this( true, prog );
	}

	protected SparkExecutionContext(boolean allocateVars, Program prog) 
	{
		//protected constructor to force use of ExecutionContextFactory
		super( allocateVars, prog );
		synchronized(SparkExecutionContext.class) {
			if(_singletonSpctx != null) {
				// Reuse the context
				_spctx = _singletonSpctx;
			}
			else {
				//create a default spark context (master, appname, etc refer to system properties
				//as given in the spark configuration or during spark-submit)
				if(MLContext._sc != null) {
					// This is when DML is called through spark shell
					// Will clean the passing of static variables later as this involves minimal change to DMLScript
					_spctx = new JavaSparkContext(MLContext._sc);
				}
				else {
					if(DMLScript.USE_LOCAL_SPARK_CONFIG) {
						// For now set 4 cores for integration testing :)
						SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("My local integration test app");
						// This is discouraged in spark but have added only for those testcase that cannot stop the context properly
						// conf.set("spark.driver.allowMultipleContexts", "true");
						conf.set("spark.ui.enabled", "false");
						// conf.set("spark.ui.port", "69389"); // some random port
						_spctx = new JavaSparkContext(conf);
					}
					else {
						_spctx = new JavaSparkContext();
					}
				}
				_singletonSpctx = _spctx;
			}
		}
	}
	
	public void close() {
		synchronized(SparkExecutionContext.class) {
			_spctx.stop();
			_singletonSpctx = null;
		}
	}
	
	public JavaSparkContext getSparkContext()
	{
		return _spctx;
	}
	
	/**
	 * This call returns an RDD handle for a given variable name. This includes 
	 * the creation of RDDs for in-memory or binary-block HDFS data. 
	 * 
	 * Spark instructions should call this for all matrix inputs except broadcast
	 * variables. 
	 * 
	 * TODO for reblock we should directly rely on the filename 
	 * 
	 * @param varname
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	@SuppressWarnings("unchecked")
	public JavaPairRDD<MatrixIndexes,MatrixBlock> getRDDHandleForVariable( String varname ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		MatrixObject mo = getMatrixObject(varname);
		
		//NOTE: MB this logic should be integrated into MatrixObject
		//However, for now we cannot assume that spark libraries are 
		//always available and hence only store generic references in 
		//matrix object while all the logic is in the SparkExecContext
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> rdd = null;
		//CASE 1: rdd already existing 
		if( mo.getRDDHandle()!=null )
		{
			rdd = (JavaPairRDD<MatrixIndexes, MatrixBlock>) mo.getRDDHandle();
		}
		//CASE 2: dirty in memory data
		else if( mo.isDirty() )
		{
			//get in-memory matrix block and parallelize it
			MatrixBlock mb = mo.acquireRead(); //pin matrix in memory
			rdd = toJavaPairRDD(_spctx, mb, (int)mo.getNumRowsPerBlock(), (int)mo.getNumColumnsPerBlock());
			mo.release(); //unpin matrix
			
			//keep rdd handle for future operations on it
			mo.setRDDHandle(rdd);
		}
		//CASE 3: non-dirty (file exists on HDFS)
		else
		{
			//parallelize hdfs-resident file
			rdd = _spctx.hadoopFile( mo.getFileName(), SequenceFileInputFormat.class, MatrixIndexes.class, MatrixBlock.class);
			rdd = rdd.mapToPair( new CopyBlockFunction() ); //cp is workaround for read bug
			
			//keep rdd handle for future operations on it
			mo.setRDDHandle(rdd);
		}
		
		return rdd;
	}
	
	/**
	 * TODO So far we only create broadcast variables but never destroy
	 * them. This is a memory leak which might lead to executor out-of-memory.
	 * However, in order to handle this, we need to keep track when broadcast 
	 * variables are no longer required.
	 * 
	 * @param varname
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public Broadcast<MatrixBlock> getBroadcastForVariable( String varname ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		MatrixObject mo = getMatrixObject(varname);
		
		Broadcast<MatrixBlock> bret = null;
		if( mo.getBroadcastHandle()!=null ) 
		{
			//reuse existing bradcast handle
			bret = (Broadcast<MatrixBlock>) mo.getBroadcastHandle();
		}
		else 
		{
			//read data into memory (no matter where it comes from)
			MatrixBlock mb = mo.acquireRead();
			bret = _spctx.broadcast(mb);
			mo.release();
		}
		
		return bret;
	}
	
	/**
	 * Keep the output rdd of spark rdd operations as meta data of matrix objects in the 
	 * symbol table.
	 * 
	 * Spark instructions should call this for all matrix outputs.
	 * 
	 * 
	 * @param varname
	 * @param rdd
	 * @throws DMLRuntimeException 
	 */
	public void setRDDHandleForVariable(String varname, JavaPairRDD<MatrixIndexes,MatrixBlock> rdd) 
		throws DMLRuntimeException
	{
		MatrixObject mo = getMatrixObject(varname);
		
		mo.setRDDHandle( rdd );
	}
	
	/**
	 * Utility method for creating an RDD out of an in-memory matrix block.
	 * 
	 * @param sc
	 * @param block
	 * @return
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 */
	public static JavaPairRDD<MatrixIndexes,MatrixBlock> toJavaPairRDD(JavaSparkContext sc, MatrixBlock src, int brlen, int bclen) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		LinkedList<Tuple2<MatrixIndexes,MatrixBlock>> list = new LinkedList<Tuple2<MatrixIndexes,MatrixBlock>>();
		
		if(    src.getNumRows() <= brlen 
		    && src.getNumColumns() <= bclen )
		{
			list.addLast(new Tuple2<MatrixIndexes,MatrixBlock>(new MatrixIndexes(1,1), src));
		}
		else
		{
			boolean sparse = src.isInSparseFormat();
			
			//create and write subblocks of matrix
			for(int blockRow = 0; blockRow < (int)Math.ceil(src.getNumRows()/(double)brlen); blockRow++)
				for(int blockCol = 0; blockCol < (int)Math.ceil(src.getNumColumns()/(double)bclen); blockCol++)
				{
					int maxRow = (blockRow*brlen + brlen < src.getNumRows()) ? brlen : src.getNumRows() - blockRow*brlen;
					int maxCol = (blockCol*bclen + bclen < src.getNumColumns()) ? bclen : src.getNumColumns() - blockCol*bclen;
					
					MatrixBlock block = new MatrixBlock(maxRow, maxCol, sparse);
						
					int row_offset = blockRow*brlen;
					int col_offset = blockCol*bclen;
	
					//copy submatrix to block
					src.sliceOperations( row_offset+1, row_offset+maxRow, 
							             col_offset+1, col_offset+maxCol, 
							             block );							
					
					//append block to sequence file
					MatrixIndexes indexes = new MatrixIndexes(blockRow+1, blockCol+1);
					list.addLast(new Tuple2<MatrixIndexes,MatrixBlock>(indexes, block));
				}
		}
		
		return sc.parallelizePairs(list);
	}
	
	/**
	 * This method is a generic abstraction for calls from the buffer pool.
	 * See toMatrixBlock(JavaPairRDD<MatrixIndexes,MatrixBlock> rdd, int numRows, int numCols);
	 * 
	 * @param rdd
	 * @param numRows
	 * @param numCols
	 * @return
	 * @throws DMLRuntimeException 
	 */
	@SuppressWarnings("unchecked")
	public static MatrixBlock toMatrixBlock(Object rdd, int rlen, int clen, int brlen, int bclen) 
		throws DMLRuntimeException
	{
		JavaPairRDD<MatrixIndexes,MatrixBlock> lrdd = (JavaPairRDD<MatrixIndexes,MatrixBlock>) rdd;
		return toMatrixBlock(lrdd, rlen, clen, brlen, bclen);
	}
	
	/**
	 * Utility method for creating a single matrix block out of an RDD. Note that this collect call
	 * might trigger execution of any pending transformation. 
	 * 
	 * TODO add a more efficient path for sparse matrices, see BinaryBlockReader.
	 * 
	 * @param rdd
	 * @param numRows
	 * @param numCols
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock toMatrixBlock(JavaPairRDD<MatrixIndexes,MatrixBlock> rdd, int rlen, int clen, int brlen, int bclen) 
		throws DMLRuntimeException
	{
		//current assumption always dense
		MatrixBlock out = new MatrixBlock(rlen, clen, false);
		List<Tuple2<MatrixIndexes,MatrixBlock>> list = rdd.collect();
		
		for( Tuple2<MatrixIndexes,MatrixBlock> keyval : list )
		{
			MatrixIndexes ix = keyval._1();
			MatrixBlock block = keyval._2();
			
			int row_offset = (int)(ix.getRowIndex()-1)*brlen;
			int col_offset = (int)(ix.getColumnIndex()-1)*bclen;
			int rows = block.getNumRows();
			int cols = block.getNumColumns();
			
			out.copy( row_offset, row_offset+rows-1, 
					  col_offset, col_offset+cols-1,
					  block, false );			
		}
		
		out.recomputeNonZeros();
		out.examSparsity();
		
		return out;
	}
	
	/**
	 * 
	 * @param rdd
	 * @param oinfo
	 */
	public static void writeRDDtoHDFS( Object rdd, String path, OutputInfo oinfo )
	{
		JavaPairRDD<MatrixIndexes,MatrixBlock> lrdd = (JavaPairRDD<MatrixIndexes,MatrixBlock>)rdd;
		lrdd.saveAsHadoopFile(path, 
				oinfo.outputKeyClass, 
				oinfo.outputValueClass, 
				oinfo.outputFormatClass);
	}
	
	/**
	 * Returns the available memory budget for broadcast variables in bytes.
	 * In detail, this takes into account the total executor memory as well
	 * as relative ratios for data and shuffle. Note, that this is a conservative
	 * estimate since both data memory and shuffle memory might not be fully
	 * utilized. 
	 * 
	 * @return
	 */
	public static double getBroadcastMemoryBudget()
	{
		if( _memExecutors < 0 || _memRatioData < 0 || _memRatioShuffle < 0 )
			analyzeSparkConfiguation();
		
		//70% of remaining free memory
		double membudget = OptimizerUtils.MEM_UTIL_FACTOR *
			              (  _memExecutors 
			               - _memExecutors*(_memRatioData+_memRatioShuffle));
		
		return membudget;
	}
	
	/**
	 * 
	 */
	public static void analyzeSparkConfiguation() 
	{
		SparkConf sconf = new SparkConf();
		
		//parse absolute executor memory
		String tmp = sconf.get("spark.executor.memory", "512m");
		if ( tmp.endsWith("g") || tmp.endsWith("G") )
			_memExecutors = Long.parseLong(tmp.substring(0,tmp.length()-1)) * 1024 * 1024 * 1024;
		else if ( tmp.endsWith("m") || tmp.endsWith("M") )
			_memExecutors = Long.parseLong(tmp.substring(0,tmp.length()-1)) * 1024 * 1024;
		else if( tmp.endsWith("k") || tmp.endsWith("K") )
			_memExecutors = Long.parseLong(tmp.substring(0,tmp.length()-1)) * 1024;
		else 
			_memExecutors = Long.parseLong(tmp.substring(0,tmp.length()-2));
		
		//get data and shuffle memory ratios (defaults not specified in job conf)
		_memRatioData = sconf.getDouble("spark.storage.memoryFraction", 0.6); //default 60%
		_memRatioShuffle = sconf.getDouble("spark.shuffle.memoryFraction", 0.2); //default 20%
	}
}
