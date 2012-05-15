package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class KahanObject extends Data {
	public double _sum;
	public double _correction;

	public KahanObject(double sum, double cor){
		super(DataType.OBJECT, ValueType.UNKNOWN);
		_sum=sum;
		_correction=cor;
	}

	public String toString()
	{
		return "("+_sum+", "+_correction+")";
	}
	public static int compare(KahanObject k1, KahanObject k2) {
		if(k1._sum!=k2._sum)
			return Double.compare(k1._sum, k2._sum);
		else 
			return Double.compare(k1._correction, k2._correction);
	}
	
	public void read(DataInput in) throws IOException
	{
		_sum=in.readDouble();
		_correction=in.readDouble();
	}
	
	public void write(DataOutput out) throws IOException
	{
		out.writeDouble(_sum);
		out.writeDouble(_correction);
	}
	
	public void set(KahanObject that)
	{
		this._sum=that._sum;
		this._correction=that._correction;
	}
	
	public void set(double s, double c)
	{
		this._sum=s;
		this._correction=c;
	}
	
	public boolean isAllZero()
	{
		return _sum==0 && _correction==0;
	}
}
