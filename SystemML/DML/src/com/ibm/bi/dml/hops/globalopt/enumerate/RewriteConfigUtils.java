/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.FormatType;
import com.ibm.bi.dml.hops.globalopt.enumerate.RewriteConfig.RewriteConfigType;
import com.ibm.bi.dml.lops.LopProperties.ExecType;

/**
 * Class that provides static utility methods, e.g. the method to determine the compatibility of two 
 * configuration parameters.
 */
public class RewriteConfigUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	public static boolean isCompatible(RewriteConfig p1, RewriteConfig p2) {
		return check(p1, p2) || check(p2, p1);
	}

	private static boolean check(RewriteConfig p1, RewriteConfig p2) {
		if(p1.isType(RewriteConfigType.BLOCK_SIZE) && p2.isType(RewriteConfigType.EXEC_TYPE)) {
			if(p2.getValue()==ExecType.MR.ordinal() && p1.getValue() == -1l)
				return false;
			if(p2.getValue()==ExecType.CP.ordinal() && p1.getValue() != -1l)
				return false;
			return true;
		}
//		if(p1 instanceof FormatParam && p2 instanceof BlockSizeParam) {
//			if((p1.getValue().equals(FormatParam.TEXT) || p1.getValue().equals(FormatParam.BINARY_CELL)) 
//					&& p2.getValue() == -1L)
//				return true;
//		}
		return false;
	}

	/**
	 * TODO: This logic needs to be handled differently. 
	 * @param config
	 * @return
	 */
	public static boolean isValidConfiguration(RewriteConfigSet config) {
		boolean isValid = false;
		
		RewriteConfig fmtParam = config.getConfigByType(RewriteConfigType.FORMAT_CHANGE);
		RewriteConfig bsParam = config.getConfigByType(RewriteConfigType.BLOCK_SIZE);
		RewriteConfig locationParam = config.getConfigByType(RewriteConfigType.EXEC_TYPE);
		
		isValid = checkForNonBlockedFormat(fmtParam, bsParam);
		if(isValid) {
			return true;
		}
		isValid = checkForBlockInMr(fmtParam, bsParam, locationParam);
		
		return isValid;
	}

	/**
	 * @param isValid
	 * @param fmtParam
	 * @param bsParam
	 * @param locationParam
	 * @return
	 */
	private static boolean checkForBlockInMr(RewriteConfig fmtParam, RewriteConfig bsParam, RewriteConfig locationParam) {
		boolean isValid = false;
		if(fmtParam == null || fmtParam.getValue()==FormatType.BINARY_BLOCK.ordinal()) {
			if(bsParam.isValue(-1) && locationParam.isValue(ExecType.CP.ordinal())) {
				isValid = true;
			} else if(!bsParam.isValue(-1) && locationParam.isValue(ExecType.MR.ordinal())) {
				isValid = true;
		    } else {
				isValid = false;
			}
		}
		return isValid;
	}

	/**
	 * @param isValid
	 * @param fmtParam
	 * @param bsParam
	 * @return
	 */
	private static boolean checkForNonBlockedFormat(RewriteConfig fmtParam, RewriteConfig bsParam) {
		boolean isValid = false;
		if(bsParam.isValue(-1) && fmtParam != null) {
			if(fmtParam.isValue(FormatType.TEXT_CELL.ordinal()) || fmtParam.isValue(FormatType.BINARY_CELL.ordinal())) {
				isValid = true;
			}
		}
		return isValid;
	}
	
	
	/**
	 * Expensive!
	 * @param c1
	 * @param c2
	 * @return
	 */
	public static boolean isCompatible(RewriteConfigSet c1, RewriteConfigSet c2) {
		
		RewriteConfig p1 = c1.getConfigByType(RewriteConfigType.BLOCK_SIZE);
		RewriteConfig p2 = c2.getConfigByType(RewriteConfigType.BLOCK_SIZE);
		
		RewriteConfig format1 = c1.getConfigByType(RewriteConfigType.FORMAT_CHANGE);
		RewriteConfig format2 = c2.getConfigByType(RewriteConfigType.FORMAT_CHANGE);
		
		return ((p1!=null && p2!=null)? p1.equals(p2) : true)
		       && format1.equals(format2);
	}
	
	public static RewriteConfigSet extractConfigurationFromHop(Hop hop, RewriteConfigSet template) {
		RewriteConfigSet extractedConfig = new RewriteConfigSet();
		for(RewriteConfig p : template.getConfigs()) {
			RewriteConfig extractedParam = PlanRewriter.extractRewriteConfigFromHop(hop, p.getType());
			extractedConfig.addConfig(extractedParam);
		}
		
		return extractedConfig;
	}
	
	
	///////////////
	// MB: Additional functionality
	
	
	/**
	 * 
	 * @param type
	 * @return
	 */
	public static RewriteConfig createInstance( RewriteConfigType type )
	{
		RewriteConfig ret = null;
		
		switch( type )
		{
			case BLOCK_SIZE:	ret = new RewriteConfigBlocksize(); break;
			case FORMAT_CHANGE:	ret = new RewriteConfigFormat();    break;
			case EXEC_TYPE:     ret = new RewriteConfigExecType();  break;
			case REPLICATION_FACTOR: ret = new RewriteConfigReplication(); break;
			case DATA_PARTITIONING: ret = new RewriteConfigPartitioning(); break;
			case VECTORIZATION: ret = new RewriteConfigVectorization(); break;
		}
		
		return ret;
	}


	/**
	 * Creates the complete set of RewriteConfigSets defined by the cross product of all
	 * instance values of all rewrite configs.
	 * 
	 * @param configurationParameters
	 */
	public static Collection<RewriteConfigSet> generateConfigCombinations(Collection<RewriteConfig> rewriteConfigs) 
	{
		Collection<RewriteConfigSet> ret = new LinkedList<RewriteConfigSet>();
		
		//initial set (see for iterative extension)
		ret.add(new RewriteConfigSet()); 
		
		//iterative extension (for each config, copy for each config val)
		for(RewriteConfig cp : rewriteConfigs) 
		{
			Collection<RewriteConfigSet> tmp = new LinkedList<RewriteConfigSet>();
			for(Integer defVal : cp.getDefinedValues())
			{
				RewriteConfig instance = RewriteConfigUtils.createInstance(cp.getType());
				instance.setValue( defVal );
				for(RewriteConfigSet c : ret)
				{
					RewriteConfigSet copy = new RewriteConfigSet(c);
					copy.addConfig(instance);
					tmp.add(copy);
				}
			}
			ret = tmp; //replace old set with extended set
		}
		
		//prune invalid rewrite config sets
		Iterator<RewriteConfigSet> iterator = ret.iterator();
		while(iterator.hasNext()) {
			RewriteConfigSet config = iterator.next();
			if(!RewriteConfigUtils.isValidConfiguration(config)) {
				iterator.remove();
			}
		}
		
		return ret;
	}
}
