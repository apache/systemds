/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashSet;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.Hop.VISIT_STATUS;

/**
 * This rewrite is a custom rewrite for JMLC in order to replace all persistent reads
 * and writes with transient reads and writes from the symbol table.
 * 
 * 
 */
public class RewriteRemovePersistentReadWrite extends HopRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private HashSet<String> _inputs = null;
	private HashSet<String> _outputs = null;
	
	public RewriteRemovePersistentReadWrite( String[] in, String[] out )
	{
		_inputs = new HashSet<String>();
		for( String var : in )
			_inputs.add( var );
		_outputs = new HashSet<String>();
		for( String var : out )
			_outputs.add( var );
	}
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots)
		throws HopsException
	{
		if( roots == null )
			return null;
		
		for( Hop h : roots ) 
			rule_RemovePersistentDataOp( h );
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root) 
		throws HopsException
	{
		if( root == null )
			return root;
		
		rule_RemovePersistentDataOp( root );
		
		return root;
	}
	
	/**
	 * 
	 * @param hop
	 */
	private void rule_RemovePersistentDataOp( Hop hop )
	{
		//check mark processed
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;
		
		//recursively process childs
		ArrayList<Hop> inputs = hop.getInput();
		for( int i=0; i<inputs.size(); i++ )
			rule_RemovePersistentDataOp( inputs.get(i) );

		//remove cast if unnecessary
		if( hop instanceof DataOp )
		{
			DataOp dop = (DataOp) hop;
			DataOpTypes dotype = dop.getDataOpType();
			
			switch( dotype ) 
			{
				case PERSISTENTREAD:
					if( _inputs.contains(dop.get_name()) )
						dop.setDataOpType(DataOpTypes.TRANSIENTREAD);
					break;
				case PERSISTENTWRITE:
					if( _outputs.contains(dop.get_name()) )
						dop.setDataOpType(DataOpTypes.TRANSIENTWRITE);
					break;
			}
		}
		
		//mark processed
		hop.set_visited( VISIT_STATUS.DONE );
	}
}
