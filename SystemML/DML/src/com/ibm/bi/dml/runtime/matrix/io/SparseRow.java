package com.ibm.bi.dml.runtime.matrix.io;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class SparseRow {

	public static final int initialCapacity=16;
	private int estimatedNzs=initialCapacity;
	private int maxNzs=Integer.MAX_VALUE;
	private int size=0;
	private double[] values=null;
	private int[] indexes=null;
	private static final Log LOG = LogFactory.getLog(SparseRow.class.getName());

	
	/**
	 * Computes the size of this {@link SparseRow} object in main memory,
	 * in bytes, as precisely as possible.  Used for caching purposes.
	 * 
	 * @return the size of this object in bytes
	 */
	public long getObjectSizeInMemory ()
	{
		long all_size = 28;
		if (values != null)
			all_size += values.length * 8;
		if (indexes != null)
			all_size += indexes.length * 4;
		return all_size;
	}

	public String toString()
	{
		String ret="";
		for(int i=0; i<size; i++)
			ret+=indexes[i]+": "+values[i]+"\t";
		return ret;
	}
/*	public SparseRow()
	{
		values=new double[initialCapacity];
		indexes=new int[initialCapacity];
	}
*/	
	public SparseRow(int estnns, int maxnns)
	{
		if(estnns>initialCapacity)
			estimatedNzs=estnns;
	//	else if(estnns<=0)//TODO
	//		estnns=maxnns;
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
	
/*	public void reset()
	{
		size=0;
	}
*/	
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
