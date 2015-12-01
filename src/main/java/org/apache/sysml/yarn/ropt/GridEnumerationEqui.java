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

package org.apache.sysml.yarn.ropt;

import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;

public class GridEnumerationEqui extends GridEnumeration
{
	
	public static final int DEFAULT_NSTEPS = 15;

	private int _nsteps = -1;
	
	public GridEnumerationEqui( ArrayList<ProgramBlock> prog, long min, long max ) 
		throws DMLRuntimeException
	{
		super(prog, min, max);
		
		_nsteps = DEFAULT_NSTEPS;
	}
	
	/**
	 * 
	 * @param steps
	 */
	public void setNumSteps( int steps )
	{
		_nsteps = steps;
	}
	
	@Override
	public ArrayList<Long> enumerateGridPoints() 
	{
		ArrayList<Long> ret = new ArrayList<Long>();
		long gap = (_max - _min) / (_nsteps-1);
		long v = _min;
		for (int i = 0; i < _nsteps; i++) {
			ret.add(v);
			v += gap;
		}
		return ret;
	}
}
