/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.api;

import java.util.ArrayList;

import com.ibm.bi.dml.parser.Expression;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.runtime.instructions.Instruction;

/**
 * The purpose of this proxy is to shield systemml internals from direct access to MLContext
 * which would try to load spark libraries and hence fail if these are not available. This
 * indirection is much more efficient than catching NoClassDefFoundErrors for every access
 * to MLContext (e.g., on each recompile).
 * 
 */
public class MLContextProxy 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static boolean _active = false;
	
	/**
	 * 
	 * @param flag
	 */
	public static void setActive(boolean flag) {
		_active = flag;
	}
	
	/**
	 * 
	 * @return
	 */
	public static boolean isActive() {
		return _active;
	}

	/**
	 * 
	 * @param tmp
	 */
	public static void performCleanupAfterRecompilation(ArrayList<Instruction> tmp) 
	{
		if(MLContext.getCurrentMLContext() != null) {
			MLContext.getCurrentMLContext().performCleanupAfterRecompilation(tmp);
		}
	}

	/**
	 * 
	 * @param source
	 * @param targetname
	 * @throws LanguageException 
	 */
	public static void setAppropriateVarsForRead(Expression source, String targetname) 
		throws LanguageException 
	{
		MLContext mlContext = MLContext.getCurrentMLContext();
		if(mlContext != null) {
			mlContext.setAppropriateVarsForRead(source, targetname);
		}
	}
	
}
