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

package org.apache.sysml.hops.globalopt.gdfgraph;

import java.util.ArrayList;

import org.apache.sysml.runtime.controlprogram.Program;

public class GDFGraph 
{
	
	
	private ArrayList<GDFNode> _roots = null;
	private Program _rtprog = null;
	
	
	public GDFGraph( Program prog, ArrayList<GDFNode> roots )
	{
		_rtprog = prog;
		_roots = roots;
	}
	
	public ArrayList<GDFNode> getGraphRootNodes()
	{
		return _roots;
	}
	
	public Program getRuntimeProgram()
	{
		return _rtprog;
	}
}
