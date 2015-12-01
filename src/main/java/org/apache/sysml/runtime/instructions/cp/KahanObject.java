/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class KahanObject extends Data 
{

	private static final long serialVersionUID = -5331529073327676437L;

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
	
	@Override
	public boolean equals( Object o ) {
		if( !(o instanceof KahanObject) )
			return false;
		KahanObject that = (KahanObject) o;
		return (_sum==that._sum && _correction==that._correction);
	}
	
	@Override
	public int hashCode() {
		throw new RuntimeException("hashCode() should never be called on instances of this class.");
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

	@Override
	public String getDebugName() {
		// TODO Auto-generated method stub
		return null;
	}
}
