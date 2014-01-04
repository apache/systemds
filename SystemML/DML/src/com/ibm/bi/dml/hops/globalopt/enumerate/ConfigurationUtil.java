/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import com.ibm.bi.dml.hops.Hop;

/**
 * Class that provides static utility methods, e.g. the method to determine the compatibility of two 
 * configuration parameters.
 */
public class ConfigurationUtil 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	public static boolean isCompatible(ConfigParam p1, ConfigParam p2) {
		return check(p1, p2) || check(p2, p1);
	}

	private static boolean check(ConfigParam p1, ConfigParam p2) {
		if(p1 instanceof BlockSizeParam && p2 instanceof LocationParam) {
			BlockSizeParam blockSize = (BlockSizeParam)p1;
			if(p2.getValue().equals(LocationParam.MR) && blockSize.getValue() == -1l)
				return false;
			if(p2.getValue().equals(LocationParam.CP) && blockSize.getValue() != -1l)
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
	public static boolean isValidConfiguration(Configuration config) {
		boolean isValid = false;
		
		ConfigParam fmtParam = config.getParamByName(FormatParam.NAME);
		ConfigParam bsParam = config.getParamByName(BlockSizeParam.NAME);
		ConfigParam locationParam = config.getParamByName(LocationParam.NAME);
		
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
	private static boolean checkForBlockInMr(ConfigParam fmtParam, ConfigParam bsParam, ConfigParam locationParam) {
		boolean isValid = false;
		if(fmtParam == null || fmtParam.getValue().equals(FormatParam.BINARY_BLOCK)) {
			if(bsParam.getValue().equals(-1L) && locationParam.getValue().equals(LocationParam.CP)) {
				isValid = true;
			} else if(!bsParam.getValue().equals(-1L) && locationParam.getValue().equals(LocationParam.MR)) {
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
	private static boolean checkForNonBlockedFormat(ConfigParam fmtParam, ConfigParam bsParam) {
		boolean isValid = false;
		if(bsParam.getValue().equals(-1L) && fmtParam != null) {
			if(fmtParam.getValue().equals(FormatParam.TEXT) || fmtParam.getValue().equals(FormatParam.BINARY_CELL)) {
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
	public static boolean isCompatible(Configuration c1, Configuration c2) {
		
		ConfigParam p1 = c1.getParamByName(BlockSizeParam.NAME);
		ConfigParam p2 = c2.getParamByName(BlockSizeParam.NAME);
		
		ConfigParam format1 = c1.getParamByName(FormatParam.NAME);
		ConfigParam format2 = c2.getParamByName(FormatParam.NAME);
		
		if(p1 != null && p2 != null) {
			long val1 = p1.getValue();
			long val2 = p2.getValue();
//			if(val1 != -1 && val2 != -1 && val1 != val2) //inputs are not allowed to have non matching block sizes
			if(val1 != val2) //inputs are not allowed to have non matching block sizes
				return false;
		}
		
		if(!format1.getValue().equals(format2.getValue())) {
			return false;
		}
		
		return true;
	}
	
	public static Configuration extractConfigurationFromHop(Hop hop, Configuration template) {
		Configuration extractedConfig = new Configuration();
		for(ConfigParam p : template.getParameters().values()) {
			ConfigParam extractedParam = p.extractParamFromHop(hop);
			extractedConfig.addParam(extractedParam);
		}
		
		return extractedConfig;
	}
	

}
