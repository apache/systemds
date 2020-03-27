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

package org.apache.sysds.runtime.controlprogram.parfor;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.sysds.runtime.instructions.cp.IntObject;

/**
 * A task is a logical group of one or multiple iterations (each iteration is assigned to exactly one task).
 * There, each single task is executed sequentially. See TaskPartitioner for how tasks are created and
 * ParWorker for how those tasks are eventually executed.
 * 
 * NOTE: (Extension possibility: group of statements) 
 * 
 */
public class Task implements Serializable
{
	
	private static final long serialVersionUID = 2815832451487164284L;
	
	public enum TaskType {
		RANGE, 
		SET
	}
	
	public static final int MAX_VARNAME_SIZE  = 256;
	public static final int MAX_TASK_SIZE     = Integer.MAX_VALUE-1; 
	
	private String _iterVar;
	private TaskType _type;
	private LinkedList<IntObject> _iterations; //each iteration is specified as an ordered set of index values
	
	public Task() {
		//default constructor for serialize
	}
	
	public Task( String iterVar, TaskType type ) {
		if( iterVar.length() > MAX_VARNAME_SIZE )
			throw new RuntimeException("Cannot create task, MAX_VARNAME_SIZE exceeded.");
		_iterVar = iterVar;
		_type = type;
		_iterations = new LinkedList<>();
	}
	
	public void addIteration( IntObject indexVal )  {
		if( size() >= MAX_TASK_SIZE )
			throw new RuntimeException("Cannot add iteration, MAX_TASK_SIZE reached.");
		_iterations.addLast( indexVal );
	}
	
	public List<IntObject> getIterations() {
		return _iterations;
	}
	
	public TaskType getType() {
		return _type;
	}
	
	public String getVarName() {
		return _iterVar;
	}
	
	public int size() {
		return _iterations.size();
	}

	@Override
	public String toString() {
		return toFormatedString();
	}

	public String toFormatedString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("task (type=");
		sb.append(_type);
		sb.append(", iterations={");
		int count=0;
		for( IntObject dat : _iterations )
		{
			if( count!=0 ) 
				sb.append(";");
			sb.append("[");
			sb.append(_iterVar);
			sb.append("=");
			sb.append(dat.getLongValue());
			sb.append("]");
			
			count++;
		}
		sb.append("})");
		return sb.toString();
	}

	public String toCompactString()
	{
		StringBuilder sb = new StringBuilder( );
		sb.append(_type);
		
		if( size() > 0 )
		{
			sb.append(".");
			sb.append(_iterVar);
			sb.append(".{");
		
			int count = 0;
			for( IntObject dat : _iterations )
			{
				if( count!=0 ) 
					sb.append(",");
				sb.append(dat.getLongValue());
				count++;
			}
			
			sb.append("}");
		}
		
		return sb.toString();
	}

	public String toCompactString( int maxDigits )
	{
		StringBuilder sb = new StringBuilder( );
		sb.append(_type);
		
		if( size() > 0 )
		{
			sb.append(".");
			sb.append(_iterVar);
			sb.append(".{");
		
			int count = 0;
			for( IntObject dat : _iterations )
			{
				if( count!=0 ) 
					sb.append(",");
				
				String tmp = String.valueOf(dat.getLongValue());
				for( int k=tmp.length(); k<maxDigits; k++ )
					sb.append("0");
				sb.append(tmp);
				count++;
			}
			
			sb.append("}");
		}
		
		return sb.toString();
	}

	public static Task parseCompactString( String stask )
	{
		StringTokenizer st = new StringTokenizer( stask.trim(), "." );
		TaskType type = TaskType.valueOf(st.nextToken());
		String meta = st.nextToken();
		Task newTask = new Task(meta, type);
		
		//iteration data
		String sdata = st.nextToken();
		sdata = sdata.substring(1,sdata.length()-1); // remove brackets
		StringTokenizer st2 = new StringTokenizer(sdata, ",");
		while( st2.hasMoreTokens() ) {
			//create new iteration
			String lsdata = st2.nextToken();
			IntObject ldata = new IntObject(Integer.parseInt(lsdata));
			newTask.addIteration(ldata);
		}
		
		return newTask;
	}
}
