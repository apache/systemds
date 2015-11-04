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

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.utils.Explain;

/**
 * This class allows to check hop dags for validity, e.g., parent-child linking.
 * It purpose is soley for debugging purposes (enabled in ProgramRewriter).
 * 
 */
public class HopDagValidator 
{
	private static final Log LOG = LogFactory.getLog(HopDagValidator.class.getName());
	
	/**
	 * 
	 * @param roots
	 * @throws HopsException 
	 * @throws DMLRuntimeException 
	 */
	public static void validateHopDag(ArrayList<Hop> roots) 
		throws HopsException
	{
		if( roots == null )
			return;
		try
		{
			Hop.resetVisitStatus(roots);
			for( Hop hop : roots )
				rValidateHop(hop);
		}
		catch(HopsException ex)
		{
			try {
				LOG.error( "\n"+Explain.explainHops(roots) );
			}catch(DMLRuntimeException e){}
			
			throw ex;
		}
	}
	
	/**
	 * 
	 * @param root
	 * @throws HopsException 
	 */
	public static void validateHopDag(Hop root) 
		throws HopsException
	{
		if( root == null )
			return;
		
		try
		{
			root.resetVisitStatus();
			rValidateHop(root);
		}
		catch(HopsException ex)
		{
			try {
				LOG.error( "\n"+Explain.explain(root) );
			}catch(DMLRuntimeException e){}
			
			throw ex;
		}
	}
	
	/**
	 * 
	 * @param hop
	 * @throws HopsException
	 */
	private static void rValidateHop( Hop hop ) 
		throws HopsException
	{
		if(hop.getVisited() == Hop.VisitStatus.DONE)
			return;
		
		//check parent linking
		for( Hop parent : hop.getParent() )
			if( !parent.getInput().contains(hop) )
				throw new HopsException("Hop id="+hop.getHopID()+" not properly linked to its parent pid="+parent.getHopID()+" "+parent.getClass().getName());
		
		//check child linking
		for( Hop child : hop.getInput() )
			if( !child.getParent().contains(hop) )
				throw new HopsException("Hop id="+hop.getHopID()+" not properly linked to its child cid="+child.getHopID()+" "+child.getClass().getName());
		
		//check empty childs
		if( hop.getInput().isEmpty() )
			if( !(hop instanceof DataOp || hop instanceof LiteralOp) )
				throw new HopsException("Hop id="+hop.getHopID()+" is not a dataop/literal but has no childs.");
		
		//recursively process childs
		for( Hop child : hop.getInput() )
			rValidateHop(child);
		
		hop.setVisited(Hop.VisitStatus.DONE);
	}
}
