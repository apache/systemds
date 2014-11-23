/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import com.ibm.bi.dml.runtime.util.SortUtils;

public class SparseRow 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//initial capacity of any created sparse row
	//WARNING: be aware that this affects the core memory estimates (incl. implicit assumptions)! 
	public static final int initialCapacity = 4;
	
	private int estimatedNzs = initialCapacity;
	private int maxNzs = Integer.MAX_VALUE;
	private int size = 0;
	private double[] values = null;
	private int[] indexes = null;

	public SparseRow(int estnns, int maxnns)
	{
		if(estnns>initialCapacity)
			estimatedNzs=estnns;
		maxNzs=maxnns;
		if(estnns<initialCapacity && estnns>0)
		{
			//LOG.trace("Allocating 1 .. " + estnns);
			values=new double[estnns];
			indexes=new int[estnns];
		}else
		{
			//LOG.trace("Allocating 2 .. " + estnns);
			values=new double[initialCapacity];
			indexes=new int[initialCapacity];
		}
	}
	
	public SparseRow(SparseRow that)
	{
		size=that.size;
		int capa=Math.max(initialCapacity, that.size);
		values=new double[capa];
		indexes=new int[capa];
		System.arraycopy(that.values, 0, values, 0, that.size);
		System.arraycopy(that.indexes, 0, indexes, 0, that.size);
	}
	
	public void truncate(int newsize)
	{
		if(newsize>size || newsize<0)
			throw new RuntimeException("truncate size: "+newsize+" should <= size: "+size+" and >=0");
		size=newsize;
	}
	
	public int size()
	{
		return size;
	}
	
	public double[] getValueContainer()
	{
		return values;
	}
	
	public int[] getIndexContainer()
	{
		return indexes;
	}
	
	public void setValueContainer(double[] d) {
		values = d;
	}
	
	public void setIndexContainer(int[] i) {
		indexes = i;
	}
	
	public int capacity()
	{
		return values.length;
	}
	
	public SparseRow(int capacity)
	{
		values=new double[capacity];
		indexes=new int[capacity];
	}
	
	public void copy(SparseRow that)
	{
		if(values.length<that.size)
			recap(that.size);
		System.arraycopy(that.values, 0, values, 0, that.size);
		System.arraycopy(that.indexes, 0, indexes, 0, that.size);
		size=that.size;
	}
	
	public void append(int col, double v)
	{
		if(v==0.0)
			return;
		if(size==values.length)
			recap();
		values[size]=v;
		indexes[size]=col;
		size++;
	}
	
	public void reset(int estnns, int maxnns)
	{
		this.estimatedNzs=estnns;
		this.maxNzs=maxnns;
		size=0;
	}
	
	public void recap(int newCap) {
		if(newCap<=values.length)
			return;
		double[] oldvalues=values;
		int[] oldindexes=indexes;
		values=new double[newCap];
		indexes=new int[newCap];
		System.arraycopy(oldvalues, 0, values, 0, size);
		System.arraycopy(oldindexes, 0, indexes, 0, size);
	}
	/*
	 * doubling before capacity reaches estimated nonzeros, then 1.1x after that
	 * for default behavor: always 1.1x
	 */
	private int newCapacity()
	{
		if(values.length<this.estimatedNzs)
		{
			//System.out.println(">> capacity change from "+values.length+" to "+Math.min(this.estimatedNzs, values.length*2)+" , est: "+estimatedNzs+", max: "+maxNzs);
			return Math.min(this.estimatedNzs, values.length*2);
		}
		else
		{
			//System.out.println(">> capacity change from "+values.length+" to "+(int) Math.min(this.maxNzs, Math.floor((double)(values.length)*1.1))+" , est: "+estimatedNzs+", max: "+maxNzs);
			return (int) Math.min(this.maxNzs, Math.ceil((double)(values.length)*1.1)); //exponential growth
			//return (int) Math.min(this.maxNzs, values.length+Math.floor((double)(estimatedNzs)*0.1)); //constant growth
		}
	}
	
	private void recap() {
		recap(newCapacity());
	}

	public boolean set(int col, double v)
	{
		int index=binarySearch(col);
		if(index<size && col==indexes[index])
		{
			values[index]=v;
			return false;
		}
		else
		{
			if(v==0.0)
				return false;
			if(size==values.length)
				resizeAndInsert(index, col, v);
			else
				shiftAndInsert(index, col, v);
			return true;
		}
	}
	
	
	/*
	 * Copies an entire sparserow into an existing sparserow in order to
	 * reduce the shifting effort.
	 * 
	 * Use case: copy sparse-sparse but currently not used since specific case
	 * for read.
	 * 
	 * @param col_offset
	 * @param arow
	 * @return
	 */
	/*
    public boolean set( int col_offset, SparseRow arow )
	{
		int index = searchIndexesFirstGT(col_offset);
		
		if( size+arow.size>values.length )
		{
			//resize and insert
			int newCap=Math.max(newCapacity(),values.length+arow.size);
			double[] oldvalues=values;
			int[] oldindexes=indexes;
			values=new double[newCap];
			indexes=new int[newCap];
			System.arraycopy(oldvalues, 0, values, 0, index);
			System.arraycopy(oldindexes, 0, indexes, 0, index);
			
			System.arraycopy(arow.values, 0, values, index, arow.size);
			for( int i=0; i<arow.size; i++ )
				indexes[index+i] = col_offset + arow.indexes[i];

			System.arraycopy(oldvalues, index, values, index+arow.size, size-index);
			System.arraycopy(oldindexes, index, indexes, index+arow.size, size-index);
			size+=arow.size;
		}
		else
		{
			//shift and insert
			System.arraycopy(values, index, values, index+arow.size, size-index);
			System.arraycopy(indexes, index, indexes, index+arow.size, size-index);			
			System.arraycopy(arow.values, 0, values, index, arow.size);
			for( int i=0; i<arow.size; i++ )
				indexes[index+i] = col_offset + arow.indexes[i];
			size+=arow.size;
		}
			
		return true;
	}
	*/
	
	private void shiftAndInsert(int index, int col, double v) {
		for(int i=size; i>index; i--)
		{
			values[i]=values[i-1];
			indexes[i]=indexes[i-1];
		}
		values[index]=v;
		indexes[index]=col;
		size++;
	}

	private void resizeAndInsert(int index, int col, double v) {
		int newCap=newCapacity();
		double[] oldvalues=values;
		int[] oldindexes=indexes;
		values=new double[newCap];
		indexes=new int[newCap];
		System.arraycopy(oldvalues, 0, values, 0, index);
		System.arraycopy(oldindexes, 0, indexes, 0, index);
		indexes[index]=col;
		values[index]=v;
		System.arraycopy(oldvalues, index, values, index+1, size-index);
		System.arraycopy(oldindexes, index, indexes, index+1, size-index);
		size++;
	}

	public double get(int col)
	{
		int index=binarySearch(col);
		if(index<size && col==indexes[index])
			return values[index];
		else
			return 0;
	}
	private int binarySearch(int x)
	{
		 int min = 0;
		 int max =size-1;
		 while(min<=max)
		 {
			 int mid=min+(max-min)/2;
			 if(x<indexes[mid])
				 max=mid-1;
			 else if(x>indexes[mid])
				 min=mid+1;
			 else
				 return mid;
		 }
		 return min;
	}
	
	public int searchIndexesFirstGTE(int col)
	{
		int index=binarySearch(col);
		if(index>=size)
			return -1;
		else return index;
	}
	
	public int searchIndexesFirstLTE(int col)
	{
		int index=binarySearch(col);
		if(index<size && col==indexes[index])
			return index;
		else if(index-1<0)
			return -1;
		else
			return index-1;
	}
	
/*	public int searchIndexesFirstLT(int col)
	{
		int index=binarySearch(col);
		if(index<size && col==indexes[index])
			return index-1;
		else
			return index;
	}*/
	
	public int searchIndexesFirstGT(int col)
	{
		int index=binarySearch(col);
		if(index<size && col==indexes[index])
			return index+1;
		else if(index>=0)
			return index;
		else
			return -1;
	}
	
	public void deleteIndexRange(int lowerIndex, int upperIndex)
	{
		int start=searchIndexesFirstGTE(lowerIndex);
		//System.out.println("start: "+start);
		if(start<0) return;
		int end=searchIndexesFirstGT(upperIndex);
		//System.out.println("end: "+end);
		if(end<0 || start>end) return;
		for(int i=0; i<size-end; i++)
		{
			indexes[start+i]=indexes[end+i];
			values[start+i]=values[end+i];
		}
		size-=(end-start);
	}
	
	public void deleteIndex( int index )
	{
		int pos=binarySearch(index);
		if(pos<size && index==indexes[pos])
		{
			for(int i=pos; i<size-1; i++)
			{
				indexes[i]=indexes[i+1];
				values[i]=values[i+1];
			}		
			size--;	
		}
	}
	
	public void deleteIndexComplementaryRange(int lowerIndex, int upperIndex)
	{
		int start=searchIndexesFirstGTE(lowerIndex);
		//System.out.println("start: "+start);
		if(start<0) return;
		int end=searchIndexesFirstGT(upperIndex);
		//System.out.println("end: "+end);
		if(end<0 || start>end) return;
		for(int i=0; i<end-start; i++)
		{
			indexes[i]=indexes[start+i];
			values[i]=values[start+i];
		}
		size=(end-start);
	}
	
	/**
	 * In-place sort of column-index value pairs in order to allow binary search
	 * after constant-time append was used for reading unordered sparse rows. We
	 * first check if already sorted and subsequently sort if necessary in order
	 * to get O(n) bestcase.
	 * 
	 * Note: In-place sort necessary in order to guarantee the memory estimate
	 * for operations that implicitly read that data set.
	 */
	public void sort()
	{
		if( size<=100 || !SortUtils.isSorted(0, size, indexes) )
			SortUtils.sort(0, size, indexes, values);
	}
	
	/**
	 * In-place compaction of non-zero-entries; removes zero entries and
	 * shifts non-zero entries to the left if necessary.
	 */
	public void compact() 
	{
		int nnz = 0;
		for( int i=0; i<size; i++ ) 
			if( values[i] != 0 ){
				values[nnz] = values[i];
				indexes[nnz] = indexes[i];
				nnz++;
			}
		size = nnz; //adjust row size
	}
	
	/**
	 * 
	 */
	@Override
	public String toString()
	{
		String ret="";
		for(int i=0; i<size; i++)
			ret+=indexes[i]+": "+values[i]+"\t";
		return ret;
	}
	
	public static void main(String[] args) throws Exception
	{
		SparseRow row=new SparseRow(6, 40);
		row.append(9, 21);
		row.append(11, 43);
		row.append(24, 23);
		row.append(30, 53);
		row.append(37, 95);
		row.append(38,38);
		
		int start=row.searchIndexesFirstGTE((int)0);
		System.out.println("start: "+start);
		if(start<0) start=row.size();
		int end=row.searchIndexesFirstGT((int)8);
		System.out.println("end: "+end);
		if(end<0) end=row.size();
	
		{
			System.out.println("----------------------");
			System.out.println("row: "+row);
			System.out.println("start: "+start);
			System.out.println("end: "+end);
		}
		
	}
}
