/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

/**
 * 
 */
public class RDDSortUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * 
	 * @param in
	 * @param rlen
	 * @param brlen
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortByVal( JavaPairRDD<MatrixIndexes, MatrixBlock> in, long rlen, int brlen )
	{
		//create value-index rdd from inputs
		JavaRDD<Double> dvals = in.values()
				.flatMap(new ExtractDoubleValuesFunction());
	
		//sort (creates sorted range per partition)
		long hdfsBlocksize = InfrastructureAnalyzer.getHDFSBlockSize();
		int numPartitions = (int)Math.ceil(((double)rlen*8)/hdfsBlocksize);
		JavaRDD<Double> sdvals = dvals
				.sortBy(new CreateDoubleKeyFunction(), true, numPartitions);
		
		//obtain partition sizes
		List<Tuple2<Integer,Long>> offsets = sdvals
				.mapPartitionsWithIndex(new GetPartitionSizesFunction(), true)
				.collect();
		
		//compute partition offsets via shifted prefix sum
		Tuple2<Integer,Long>[] tmpoffsets = offsets.toArray(new Tuple2[0]);
		long[][] poffsets = new long[tmpoffsets.length][2];
		poffsets[0] = new long[]{tmpoffsets[0]._1,0};
		for( int i=1; i<poffsets.length; i++ ) {
			poffsets[i][0] = tmpoffsets[i]._1;
			poffsets[i][1] = poffsets[i-1][1]+tmpoffsets[i]._2;
		}
		
		//create binary block rdd (offsets not shipped via broadcast since only used once
		//and in order to simplify the api of these sort utils)		
		JavaPairRDD<MatrixIndexes, MatrixBlock> ret = sdvals
				.mapPartitionsWithIndex(new ConvertToBinaryBlockFunction(poffsets, rlen, brlen), true)
				.mapToPair(new UnfoldBinaryBlockFunction());
		ret = RDDAggregateUtils.mergeByKey(ret);		
		
		return ret;
	}
	
	/**
	 * 
	 * @param in
	 * @param in2
	 * @param rlen
	 * @param brlen
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortByVal( JavaPairRDD<MatrixIndexes, MatrixBlock> in, 
			JavaPairRDD<MatrixIndexes, MatrixBlock> in2, long rlen, int brlen )
	{
		//create value-index rdd from inputs
		JavaRDD<DoublePair> dvals = in.join(in2).values()
				.flatMap(new ExtractDoubleValuesFunction2());
	
		//sort (creates sorted range per partition)
		long hdfsBlocksize = InfrastructureAnalyzer.getHDFSBlockSize();
		int numPartitions = (int)Math.ceil(((double)rlen*8)/hdfsBlocksize);
		JavaRDD<DoublePair> sdvals = dvals
				.sortBy(new CreateDoubleKeyFunction2(), true, numPartitions);
		
		//obtain partition sizes
		List<Tuple2<Integer,Long>> offsets = sdvals
				.mapPartitionsWithIndex(new GetPartitionSizesFunction2(), true)
				.collect();
		
		//compute partition offsets via shifted prefix sum
		Tuple2<Integer,Long>[] tmpoffsets = offsets.toArray(new Tuple2[0]);
		long[][] poffsets = new long[tmpoffsets.length][2];
		poffsets[0] = new long[]{tmpoffsets[0]._1,0};
		for( int i=1; i<poffsets.length; i++ ) {
			poffsets[i][0] = tmpoffsets[i]._1;
			poffsets[i][1] = poffsets[i-1][1]+tmpoffsets[i]._2;
		}
		
		//create binary block rdd (offsets not shipped via broadcast since only used once
		//and in order to simplify the api of these sort utils)		
		JavaPairRDD<MatrixIndexes, MatrixBlock> ret = sdvals
				.mapPartitionsWithIndex(new ConvertToBinaryBlockFunction2(poffsets, rlen, brlen), true)
				.mapToPair(new UnfoldBinaryBlockFunction());
		ret = RDDAggregateUtils.mergeByKey(ret);		
		
		return ret;
	}
	
	/**
	 * 
	 * @param in
	 * @param rlen
	 * @param brlen
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortIndexesByVal( JavaPairRDD<MatrixIndexes, MatrixBlock> in, 
			boolean asc, long rlen, int brlen )
	{
		//create value-index rdd from inputs
		JavaPairRDD<ValueIndexPair, Double> dvals = in
				.flatMapToPair(new ExtractDoubleValuesWithIndexFunction(brlen));
	
		//sort (creates sorted range per partition)
		long hdfsBlocksize = InfrastructureAnalyzer.getHDFSBlockSize();
		int numPartitions = (int)Math.ceil(((double)rlen*16)/hdfsBlocksize);
		JavaRDD<ValueIndexPair> sdvals = dvals
				.sortByKey(new IndexComparator(asc), true, numPartitions)
				.keys(); //workaround for index comparator
		
		//obtain partition sizes
		List<Tuple2<Integer,Long>> offsets = sdvals
				.mapPartitionsWithIndex(new GetPartitionSizesFunction3(), true)
				.collect();
		
		//compute partition offsets via shifted prefix sum
		Tuple2<Integer,Long>[] tmpoffsets = offsets.toArray(new Tuple2[0]);
		long[][] poffsets = new long[tmpoffsets.length][2];
		poffsets[0] = new long[]{tmpoffsets[0]._1,0};
		for( int i=1; i<poffsets.length; i++ ) {
			poffsets[i][0] = tmpoffsets[i]._1;
			poffsets[i][1] = poffsets[i-1][1]+tmpoffsets[i]._2;
		}
		
		//create binary block rdd (offsets not shipped via broadcast since only used once
		//and in order to simplify the api of these sort utils)		
		JavaPairRDD<MatrixIndexes, MatrixBlock> ret = sdvals
				.mapPartitionsWithIndex(new ConvertToBinaryBlockFunction3(poffsets, rlen, brlen), true)
				.mapToPair(new UnfoldBinaryBlockFunction());
		ret = RDDAggregateUtils.mergeByKey(ret);		
		
		return ret;	
	}
	
	/**
	 * 
	 */
	private static class ExtractDoubleValuesFunction implements FlatMapFunction<MatrixBlock,Double> 
	{
		private static final long serialVersionUID = 6888003502286282876L;

		@Override
		public Iterable<Double> call(MatrixBlock arg0) 
			throws Exception 
		{
			return DataConverter.convertToDoubleList(arg0);
		}		
	}

	/**
	 * 
	 */
	private static class ExtractDoubleValuesFunction2 implements FlatMapFunction<Tuple2<MatrixBlock,MatrixBlock>,DoublePair> 
	{
		private static final long serialVersionUID = 2132672563825289022L;

		@Override
		public Iterable<DoublePair> call(Tuple2<MatrixBlock,MatrixBlock> arg0) 
			throws Exception 
		{
			ArrayList<DoublePair> ret = new ArrayList<DoublePair>(); 
			MatrixBlock mb1 = arg0._1();
			MatrixBlock mb2 = arg0._2();
			
			for( int i=0; i<mb1.getNumRows(); i++) {
				ret.add(new DoublePair(
						mb1.quickGetValue(i, 0),
						mb2.quickGetValue(i, 0)));
			}
			
			return ret;
		}		
	}
	
	private static class ExtractDoubleValuesWithIndexFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>,ValueIndexPair,Double> 
	{
		private static final long serialVersionUID = -3976735381580482118L;
		
		private int _brlen = -1;
		
		public ExtractDoubleValuesWithIndexFunction(int brlen)
		{
			_brlen = brlen;
		}
		
		@Override
		public Iterable<Tuple2<ValueIndexPair,Double>> call(Tuple2<MatrixIndexes,MatrixBlock> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<ValueIndexPair,Double>> ret = new ArrayList<Tuple2<ValueIndexPair,Double>>(); 
			MatrixIndexes ix = arg0._1();
			MatrixBlock mb = arg0._2();
			
			long ixoffset = (ix.getRowIndex()-1)*_brlen;
			for( int i=0; i<mb.getNumRows(); i++) {
				double val = mb.quickGetValue(i, 0);
				ret.add(new Tuple2<ValueIndexPair,Double>(
						new ValueIndexPair(val,ixoffset+i+1), val));
			}
			
			return ret;
		}		
	}
	
	/**
	 * 
	 */
	private static class CreateDoubleKeyFunction implements Function<Double,Double> 
	{
		private static final long serialVersionUID = 2021786334763247835L;

		@Override
		public Double call(Double arg0) 
			throws Exception 
		{
			return arg0;
		}		
	}
	
	/**
	 * 
	 */
	private static class CreateDoubleKeyFunction2 implements Function<DoublePair,Double> 
	{
		private static final long serialVersionUID = -7954819651274239592L;

		@Override
		public Double call(DoublePair arg0) 
			throws Exception 
		{
			return arg0.val1;
		}		
	}
	
	/**
	 * 
	 */
	private static class GetPartitionSizesFunction implements Function2<Integer,Iterator<Double>,Iterator<Tuple2<Integer,Long>>> 
	{
		private static final long serialVersionUID = 5491422398886449135L;

		@Override
		@SuppressWarnings("unchecked")
		public Iterator<Tuple2<Integer,Long>> call(Integer arg0, Iterator<Double> arg1)
			throws Exception 
		{
			ArrayList<Tuple2<Integer,Long>> ret = new ArrayList<Tuple2<Integer,Long>>();
			
			if( arg1 instanceof Collection ) {
				ret.add(new Tuple2<Integer,Long>(arg0,(long)((Collection<Double>)arg1).size()));
			}
			else {
				int size = 0;
				while( arg1.hasNext() ) {
					arg1.next();
					size++;
				}
				ret.add(new Tuple2<Integer,Long>(arg0,(long)size));
			}
			
			return ret.iterator();
		}
	}
	
	/**
	 * 
	 */
	private static class GetPartitionSizesFunction2 implements Function2<Integer,Iterator<DoublePair>,Iterator<Tuple2<Integer,Long>>> 
	{
		private static final long serialVersionUID = 3735593665863933572L;

		@Override
		@SuppressWarnings("unchecked")
		public Iterator<Tuple2<Integer,Long>> call(Integer arg0, Iterator<DoublePair> arg1)
			throws Exception 
		{
			ArrayList<Tuple2<Integer,Long>> ret = new ArrayList<Tuple2<Integer,Long>>();
			
			if( arg1 instanceof Collection ) {
				ret.add(new Tuple2<Integer,Long>(arg0,(long)((Collection<DoublePair>)arg1).size()));
			}
			else {
				int size = 0;
				while( arg1.hasNext() ) {
					arg1.next();
					size++;
				}
				ret.add(new Tuple2<Integer,Long>(arg0,(long)size));
			}
			
			return ret.iterator();
		}
	}
	
	/**
	 * 
	 */
	private static class GetPartitionSizesFunction3 implements Function2<Integer,Iterator<ValueIndexPair>,Iterator<Tuple2<Integer,Long>>> 
	{
		private static final long serialVersionUID = -6144308545615151619L;

		@Override
		@SuppressWarnings("unchecked")
		public Iterator<Tuple2<Integer,Long>> call(Integer arg0, Iterator<ValueIndexPair> arg1)
			throws Exception 
		{
			ArrayList<Tuple2<Integer,Long>> ret = new ArrayList<Tuple2<Integer,Long>>();
			
			if( arg1 instanceof Collection ) {
				ret.add(new Tuple2<Integer,Long>(arg0,(long)((Collection<ValueIndexPair>)arg1).size()));
			}
			else {
				int size = 0;
				while( arg1.hasNext() ) {
					arg1.next();
					size++;
				}
				ret.add(new Tuple2<Integer,Long>(arg0,(long)size));
			}
			
			return ret.iterator();
		}
	}
	/**
	 * 
	 */
	private static class ConvertToBinaryBlockFunction implements Function2<Integer,Iterator<Double>,Iterator<Tuple2<MatrixIndexes,MatrixBlock>>> 
	{
		private static final long serialVersionUID = 5000298196472931653L;
		
		private long[][] _offsets = null; //id/offset
		private long _rlen = -1;
		private int _brlen = -1;
		
		public ConvertToBinaryBlockFunction(long[][] boff, long rlen, int brlen)
		{
			_offsets = boff;
			_rlen = rlen;
			_brlen = brlen;
		}
		
		public Iterator<Tuple2<MatrixIndexes,MatrixBlock>> call(Integer arg0, Iterator<Double> arg1)
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			long rowoffset = getOffset(arg0)+1; //1-based
			
			//create initial matrix block
			long rix = UtilFunctions.blockIndexCalculation(rowoffset, _brlen);
			long len = UtilFunctions.computeBlockSize(_rlen, rix, _brlen);
			int pos = UtilFunctions.cellInBlockCalculation(rowoffset, _brlen);
			MatrixIndexes ix = new MatrixIndexes(rix,1);
			MatrixBlock mb = new MatrixBlock((int)len, 1, false);
			
			while( arg1.hasNext() ) 
			{
				mb.quickSetValue(pos, 0, arg1.next());
				pos++;
				
				//create next block if necessary
				if( pos==mb.getNumRows() ){
					ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(ix,mb));
					rix++;
					len = UtilFunctions.computeBlockSize(_rlen, rix, _brlen);
					ix = new MatrixIndexes(rix,1);
					mb = new MatrixBlock((int)len, 1, false);
					pos=0;
				}
			}
			
			//flush last block
			if( pos != 0 )
				ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(ix,mb));
			
			return ret.iterator();
		}
		
		private long getOffset(long partid) 
			throws DMLRuntimeException
		{
			for( int i=0; i<_offsets.length; i++ )
				if( _offsets[i][0] == partid )
					return _offsets[i][1];
				
			throw new DMLRuntimeException("Could not find partition offset for id="+partid);	
		}
	}

	/**
	 * 
	 */
	private static class ConvertToBinaryBlockFunction2 implements Function2<Integer,Iterator<DoublePair>,Iterator<Tuple2<MatrixIndexes,MatrixBlock>>> 
	{
		private static final long serialVersionUID = -8638434373377180192L;
		
		private long[][] _offsets = null; //id/offset
		private long _rlen = -1;
		private int _brlen = -1;
		
		public ConvertToBinaryBlockFunction2(long[][] boff, long rlen, int brlen)
		{
			_offsets = boff;
			_rlen = rlen;
			_brlen = brlen;
		}
		
		public Iterator<Tuple2<MatrixIndexes,MatrixBlock>> call(Integer arg0, Iterator<DoublePair> arg1)
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			long rowoffset = getOffset(arg0)+1; //1-based
			
			//create initial matrix block
			long rix = UtilFunctions.blockIndexCalculation(rowoffset, _brlen);
			long len = UtilFunctions.computeBlockSize(_rlen, rix, _brlen);
			int pos = UtilFunctions.cellInBlockCalculation(rowoffset, _brlen);
			MatrixIndexes ix = new MatrixIndexes(rix,1);
			MatrixBlock mb = new MatrixBlock((int)len, 2, false);
			
			while( arg1.hasNext() ) 
			{
				DoublePair val = arg1.next();
				mb.quickSetValue(pos, 0, val.val1);
				mb.quickSetValue(pos, 1, val.val2);
				pos++;
				
				//create next block if necessary
				if( pos==mb.getNumRows() ){
					ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(ix,mb));
					rix++;
					len = UtilFunctions.computeBlockSize(_rlen, rix, _brlen);
					ix = new MatrixIndexes(rix,1);
					mb = new MatrixBlock((int)len, 2, false);
					pos=0;
				}
			}
			
			//flush last block
			if( pos != 0 )
				ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(ix,mb));
			
			return ret.iterator();
		}
		
		private long getOffset(long partid) 
			throws DMLRuntimeException
		{
			for( int i=0; i<_offsets.length; i++ )
				if( _offsets[i][0] == partid )
					return _offsets[i][1];
				
			throw new DMLRuntimeException("Could not find partition offset for id="+partid);	
		}
	}
	
	/**
	 * 
	 */
	private static class ConvertToBinaryBlockFunction3 implements Function2<Integer,Iterator<ValueIndexPair>,Iterator<Tuple2<MatrixIndexes,MatrixBlock>>> 
	{
		private static final long serialVersionUID = -8638434373377180192L;
		
		private long[][] _offsets = null; //id/offset
		private long _rlen = -1;
		private int _brlen = -1;
		
		public ConvertToBinaryBlockFunction3(long[][] boff, long rlen, int brlen)
		{
			_offsets = boff;
			_rlen = rlen;
			_brlen = brlen;
		}
		
		public Iterator<Tuple2<MatrixIndexes,MatrixBlock>> call(Integer arg0, Iterator<ValueIndexPair> arg1)
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			long rowoffset = getOffset(arg0)+1; //1-based
			
			//create initial matrix block
			long rix = UtilFunctions.blockIndexCalculation(rowoffset, _brlen);
			long len = UtilFunctions.computeBlockSize(_rlen, rix, _brlen);
			int pos = UtilFunctions.cellInBlockCalculation(rowoffset, _brlen);
			MatrixIndexes ix = new MatrixIndexes(rix,1);
			MatrixBlock mb = new MatrixBlock((int)len, 1, false);
			
			while( arg1.hasNext() ) 
			{
				ValueIndexPair val = arg1.next();
				mb.quickSetValue(pos, 0, val.ix);
				pos++;
				
				//create next block if necessary
				if( pos==mb.getNumRows() ){
					ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(ix,mb));
					rix++;
					len = UtilFunctions.computeBlockSize(_rlen, rix, _brlen);
					ix = new MatrixIndexes(rix,1);
					mb = new MatrixBlock((int)len, 1, false);
					pos=0;
				}
			}
			
			//flush last block
			if( pos != 0 )
				ret.add(new Tuple2<MatrixIndexes,MatrixBlock>(ix,mb));
			
			return ret.iterator();
		}
		
		private long getOffset(long partid) 
			throws DMLRuntimeException
		{
			for( int i=0; i<_offsets.length; i++ )
				if( _offsets[i][0] == partid )
					return _offsets[i][1];
				
			throw new DMLRuntimeException("Could not find partition offset for id="+partid);	
		}
	}

	
	/**
	 * 
	 */
	private static class UnfoldBinaryBlockFunction implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = -5509821097041916225L;

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> t) 
			throws Exception 
		{
			return t;
		}
	}
	
	/**
	 * More memory-efficient representation than Tuple2<Double,Double> which requires
	 * three instead of one object per cell.
	 */
	private static class DoublePair implements Serializable
	{
		private static final long serialVersionUID = 4373356163734559009L;
		
		public double val1;
		public double val2;
		
		public DoublePair(double d1, double d2) {
			val1 = d1;
			val2 = d2;
		}
	}
	
	/**
	 * 
	 */
	private static class ValueIndexPair implements Serializable 
	{
		private static final long serialVersionUID = -3273385845538526829L;
		
		public double val; 
		public long ix; 

		public ValueIndexPair(double dval, long lix) {
			val = dval;
			ix = lix;
		}
	}
	
	public static class IndexComparator implements Comparator<ValueIndexPair>, Serializable 
	{
		private static final long serialVersionUID = 5154839870549241343L;
		
		private boolean _asc;
		public IndexComparator(boolean asc) {
			_asc = asc;
		}
			
		@Override
		public int compare(ValueIndexPair o1, ValueIndexPair o2) 
		{
			//note: use conversion to Double and Long instead of native
			//compare for compatibility with jdk 6
			int retVal = Double.valueOf(o1.val).compareTo(o2.val);
			if(retVal != 0) {
				return (_asc ? retVal : -1*retVal);
			}
			else {
				//for stable sort
				return Long.valueOf(o1.ix).compareTo(o2.ix);
			}
		}
		
	}
}
