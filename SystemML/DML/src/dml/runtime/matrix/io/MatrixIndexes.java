package dml.runtime.matrix.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

import dml.runtime.util.UtilFunctions;

public class MatrixIndexes implements WritableComparable<MatrixIndexes>{

	private long row=-1;
	private long column=-1;
	public static final int BYTE_SIZE=(Long.SIZE+Long.SIZE)/8;
	public static long ADD_PRIME1=99991;
	public static long ADD_PRIME2=853;
	public static int DIVIDE_PRIME=51473;
	
//	private static final Log LOG = LogFactory.getLog(MatrixCellIndexes.class);
	
	public MatrixIndexes(){};
	public MatrixIndexes(long r, long c)
	{
		setIndexes(r,c);
	}
	
	public MatrixIndexes(MatrixIndexes indexes) {
		setIndexes(indexes.row, indexes.column);
	}
	public long getRowIndex()
	{
		return row;
	}
	public long getColumnIndex()
	{
		return column;
	}
	
	public void setIndexes(long r, long c)
	{
		row=r;
		column=c;
	}
	
	public void setIndexes(MatrixIndexes that) {
		
		this.row=that.row;
		this.column=that.column;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		row=in.readLong();
		column=in.readLong();
		
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong(row);
		out.writeLong(column);
		
	}
	public int compareTo(MatrixIndexes other)
	{
		if(this.row!=other.row)
			return (this.row>other.row? 1:-1);
		else if(this.column!=other.column)
			return (this.column>other.column? 1:-1);
		return 0;
	}

	public boolean equals(MatrixIndexes other)
	{
	//	LOG.info("calling equals for MatrixCellIndexes!");
		return (this.row==other.row && this.column==other.column);
	}
	
	public boolean equals(Object other)
	{
	//	LOG.info("calling equals for MatrixCellIndexes!");
		if( !(other instanceof MatrixIndexes))
			return false;
		return (this.row==((MatrixIndexes)other).row && this.column==((MatrixIndexes)other).column);
	}
	
	 public int hashCode() {
		 return UtilFunctions.longHashFunc((row<<32)+column+ADD_PRIME1)%DIVIDE_PRIME;
	 }
	
	/** A Comparator optimized for Tagged. */ 
	public static class Comparator implements RawComparator<MatrixIndexes>
	{
		@Override
		public int compare(byte[] b1, int s1, int l1,
                byte[] b2, int s2, int l2)
		{
			assert l1 == l2;
			long v1 = WritableComparator.readLong(b1, s1);
		    long v2 = WritableComparator.readLong(b2, s2);
		    if(v1!=v2)
		    	return v1<v2 ? -1 : 1;
		    v1 = WritableComparator.readLong(b1, s1+Long.SIZE/8);
		    v2 = WritableComparator.readLong(b2, s2+Long.SIZE/8);
		    return (v1<v2 ? -1 : (v1==v2 ? 0 : 1));
		}

		@Override
		public int compare(MatrixIndexes m1, MatrixIndexes m2) {
			return m1.compareTo(m2);
		}
		
	}

	public void print() {
		System.out.println("("+row+", "+column+")");
	}
	
	public String toString()
	{
		return "("+row+", "+column+")";
	}
	public int compareWithOrder(MatrixIndexes other, boolean leftcached) {
		if(!leftcached)
			return compareTo(other);
		
		if(this.column!=other.column)
			return (this.column>other.column? 1:-1);
		else if(this.row!=other.row)
			return (this.row>other.row? 1:-1);
		return 0;
	}
}
