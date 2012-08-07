package com.ibm.bi.dml.runtime.matrix.io;

public class SparseRow {

	public static final int defaultCapacity=16;
	private int size=0;
	private double[] values=null;
	private int[] indexes=null;
	
	/**
	 * Computes the size of this {@link SparseRow} object in main memory,
	 * in bytes, as precisely as possible.  Used for caching purposes.
	 * 
	 * @return the size of this object in bytes
	 */
	public long getObjectSizeInMemory ()
	{
		long all_size = 16;
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
	public SparseRow()
	{
		values=new double[defaultCapacity];
		indexes=new int[defaultCapacity];
	}
	
	public SparseRow(SparseRow that)
	{
		size=that.size;
		int capa=Math.max(defaultCapacity, that.size);
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
	
	public void reset()
	{
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
	
	private void recap() {
		
		double[] oldvalues=values;
		int[] oldindexes=indexes;
		values=new double[size+defaultCapacity];
		indexes=new int[size+defaultCapacity];
		System.arraycopy(oldvalues, 0, values, 0, size);
		System.arraycopy(oldindexes, 0, indexes, 0, size);
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
		double[] oldvalues=values;
		int[] oldindexes=indexes;
		values=new double[size+defaultCapacity];
		indexes=new int[size+defaultCapacity];
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
		SparseRow row=new SparseRow();
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
