/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.controlprogram.parfor.util;

/**
 * Helper class for representing text cell and binary cell records in order to
 * allow for buffering and buffered read/write.
 * 
 * NOTE: could be replaced by IJV.class but used in order to ensure independence.
 */
public class Cell 
{

	
	private long _row;
	private long _col;
	private double _value;
	
	public Cell( long r, long c, double v )
	{
		_row = r;
		_col = c;
		_value = v;
	}
	
	public long getRow()
	{
		return _row;
	}
	
	public long getCol()
	{
		return _col;
	}
	
	public double getValue()
	{
		return _value;
	}
	
	public void setRow( long row )
	{
		_row = row;
	}
	
	public void setCol( long col )
	{
		_col = col;
	}
	
	public void setValue( double value )
	{
		_value = value;
	}
}
