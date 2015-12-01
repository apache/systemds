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

import org.apache.sysml.hops.Hop;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;

/**
 * Crossblock operators represent 
 * 
 */
public class GDFCrossBlockNode extends GDFNode
{
	
	
	public enum CrossBlockNodeType {
		PLAIN,
		MERGE,
	}
	
	private CrossBlockNodeType _cbtype = null;
	private String _name = null;
	
	/**
	 * Constructor PLAIN crossblocknode
	 * 
	 * @param hop
	 * @param pb
	 * @param input
	 * @param name
	 */
	public GDFCrossBlockNode( Hop hop, ProgramBlock pb, GDFNode input, String name )
	{
		super(hop, pb, null);
		_type = NodeType.CROSS_BLOCK_NODE;
		_inputs = new ArrayList<GDFNode>();
		_inputs.add( input );
		
		_cbtype = CrossBlockNodeType.PLAIN;
		_name = name;
	}
	
	/**
	 * Constructor MERGE crossblocknode
	 * 
	 * @param hop
	 * @param pb
	 * @param input1
	 * @param input2
	 * @param name
	 */
	public GDFCrossBlockNode( Hop hop, ProgramBlock pb, GDFNode input1, GDFNode input2, String name )
	{
		super(hop, pb, null);
		_type = NodeType.CROSS_BLOCK_NODE;
		_inputs = new ArrayList<GDFNode>();
		_inputs.add( input1 );
		_inputs.add( input2 );
		
		_cbtype = CrossBlockNodeType.MERGE;
		_name = name;
	}

	public String getName()
	{
		return _name;
	}
	
	@Override
	public String explain(String deps) 
	{
		String ldeps = (deps!=null) ? deps : "";
		
		return "CBNode "+ldeps+" ["+_name+", "+_cbtype.toString().toLowerCase()+"]";
	}
}
