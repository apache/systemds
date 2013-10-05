/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.cost;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;

public class CostEstimationWrapper 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum CostType { 
		NUM_MRJOBS, //based on number of MR jobs, [number MR jobs]
		STATIC, // based on FLOPS, read/write, etc, [time in sec]      
		DYNAMIC // based on dynamic offline performance profile, [time in sec]
	};
	
	private static final boolean LDEBUG = false; //internal local debug level
	private static final Log LOG = LogFactory.getLog(CostEstimationWrapper.class.getName());
	private static final CostType DEFAULT_COSTTYPE = CostType.STATIC;
	
	private static CostEstimator _costEstim = null;
	
	
	static 
	{	
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("com.ibm.bi.dml.hops.cost")
				  .setLevel((Level) Level.DEBUG);
		}
		
		//create cost estimator
		try
		{
			//TODO config parameter?
			_costEstim = createCostEstimator(DEFAULT_COSTTYPE);
		}
		catch(Exception ex)
		{
			LOG.error("Failed cost estimator initialization.", ex);
		}
	}
	
	/**
	 * 
	 * @param rtprog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public static double getTimeEstimate(Program rtprog, ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		Timing time = new Timing();
		time.start();
		
		HashMap<String,VarStats> stats = new HashMap<String, VarStats>();		
		LocalVariableMap vars = (ec!=null)? ec.getVariables() : new LocalVariableMap(); 
		
		double costs = _costEstim.getTimeEstimate(rtprog, vars, stats);
		LOG.debug("Finished estimation in "+time.stop()+"ms.");
		return costs;
	}
	
	/**
	 * 
	 * @param hops
	 * @param ec
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws IOException
	 */
	public static double getTimeEstimate( ArrayList<Hop> hops, ExecutionContext ec ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException, HopsException, LopsException, IOException
	{
		Timing time = new Timing();
		time.start();
		
		HashMap<String,VarStats> stats = new HashMap<String, VarStats>();
		LocalVariableMap vars = (ec!=null)? ec.getVariables() : new LocalVariableMap(); 
		
		double costs = _costEstim.getTimeEstimate(hops, vars, stats);
		LOG.debug("Finished estimation in "+time.stop()+"ms.");
		
		return costs;
	}
	
	/**
	 * 
	 * @param type
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static CostEstimator createCostEstimator( CostType type ) 
		throws DMLRuntimeException
	{
		switch( type )
		{
			case NUM_MRJOBS:
				return new CostEstimatorNumMRJobs();
			case STATIC:
				return new CostEstimatorStaticRuntime();
			default:
				throw new DMLRuntimeException("Unknown cost type: "+type);
		}
	}	
}
