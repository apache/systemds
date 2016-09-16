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

package org.apache.sysml.conf;

import java.util.HashMap;

import org.apache.sysml.hops.OptimizerUtils;

/**
 * Basic wrapper for all compiler configurations that are configured
 * dynamically on a per script invocation basis. This allows us to
 * provide thread-local compiler configurations to prevent side-effects
 * between multiple scripts running in the same JVM process.
 * 
 */
public class CompilerConfig 
{
	public enum ConfigType {
		//Configured compiler optimization level (see OptimizerUtils for defails)
		OPT_LEVEL,
		//Configured or automatically determined binary matrix blocksize
		BLOCK_SIZE,
		//Enables parallel read/write of text (textcell, csv, mm) and binary formats
		PARALLEL_CP_READ_TEXTFORMATS,
		PARALLEL_CP_WRITE_TEXTFORMATS,
		PARALLEL_CP_READ_BINARYFORMATS,
		PARALLEL_CP_WRITE_BINARYFORMATS,
		//Enables multi-threaded operations for mm, mmchain, and tsmm, rand, wdivmm, 
		//wsloss, wumm, wcemm, uagg, tak, and groupedaggregate.
		PARALLEL_CP_MATRIX_OPERATIONS,
		//Enables multi-threaded local or distributed remote parfor operators. Otherwise 
		//parfor is restricted to parfor local with par=1.
		PARALLEL_LOCAL_OR_REMOTE_PARFOR,
		//Enables dynamic re-compilation of lops/instructions. If enabled, we recompile 
		//each program block that contains at least one hop that requires re-compilation 
		//(e.g., unknown statistics during compilation, or program blocks in functions).  
		ALLOW_DYN_RECOMPILATION,
		ALLOW_PARALLEL_DYN_RECOMPILATION,
		//Enables to put operations with data-dependent output size into individual 
		//statement blocks / program blocks. Since recompilation is done on the granularity 
		//of program blocks this enables recompilation of subsequent operations according
		//to the actual output size. This rewrite might limit the opportunity for piggybacking 
		//and therefore should only be applied if dyanmic recompilation is enabled as well.
		ALLOW_INDIVIDUAL_SB_SPECIFIC_OPS,
		//Enables common subexpression elimination in dags for persistent reads based on 
		//filenames and other relevant read meta data. Disabled for jmlc to allow binding of 
		//in-memory objects without specifying read properties.
		ALLOW_CSE_PERSISTENT_READS,
		
		//Global parser configuration (dml/pydml) to skip errors on unspecified args 
		// (modified by mlcontext / jmlc)
		IGNORE_UNSPECIFIED_ARGS, 
		//Data expression configuration (modified by mlcontext, jmlc apis); no read of meta 
		//data on mlcontext (local) /jmlc (global); ignore unknowns on jmlc
		IGNORE_READ_WRITE_METADATA, // global skip meta data reads
		REJECT_READ_WRITE_UNKNOWNS, // ignore missing meta data	
		MLCONTEXT // execution via new MLContext
	}
	
	//default flags (exposed for testing purposes only)
	public static boolean FLAG_DYN_RECOMPILE = true;
	public static boolean FLAG_PARREADWRITE_TEXT = true;
	public static boolean FLAG_PARREADWRITE_BINARY = true;
	
	private HashMap<ConfigType, Boolean> _bmap = null;
	private HashMap<ConfigType, Integer> _imap = null;
	
	public CompilerConfig() {
		_bmap = new HashMap<ConfigType, Boolean>();
		_bmap.put(ConfigType.PARALLEL_CP_READ_TEXTFORMATS, FLAG_PARREADWRITE_TEXT);
		_bmap.put(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS, FLAG_PARREADWRITE_TEXT);
		_bmap.put(ConfigType.PARALLEL_CP_READ_BINARYFORMATS, FLAG_PARREADWRITE_BINARY);
		_bmap.put(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS, FLAG_PARREADWRITE_BINARY);
		_bmap.put(ConfigType.PARALLEL_CP_MATRIX_OPERATIONS, true);
		_bmap.put(ConfigType.PARALLEL_LOCAL_OR_REMOTE_PARFOR, true);
		_bmap.put(ConfigType.ALLOW_DYN_RECOMPILATION,          FLAG_DYN_RECOMPILE);
		_bmap.put(ConfigType.ALLOW_PARALLEL_DYN_RECOMPILATION, FLAG_DYN_RECOMPILE);
		_bmap.put(ConfigType.ALLOW_INDIVIDUAL_SB_SPECIFIC_OPS, FLAG_DYN_RECOMPILE);
		_bmap.put(ConfigType.ALLOW_CSE_PERSISTENT_READS, true);
		_bmap.put(ConfigType.IGNORE_UNSPECIFIED_ARGS, false);
		_bmap.put(ConfigType.IGNORE_READ_WRITE_METADATA, false);
		_bmap.put(ConfigType.REJECT_READ_WRITE_UNKNOWNS, true);
		_bmap.put(ConfigType.MLCONTEXT, false);
		
		_imap = new HashMap<CompilerConfig.ConfigType, Integer>();
		_imap.put(ConfigType.BLOCK_SIZE, OptimizerUtils.DEFAULT_BLOCKSIZE);
		_imap.put(ConfigType.OPT_LEVEL, OptimizerUtils.DEFAULT_OPTLEVEL.ordinal());
	}
	
	@SuppressWarnings("unchecked")
	public CompilerConfig( CompilerConfig conf ) {
		_bmap = (HashMap<ConfigType, Boolean>) conf._bmap.clone();
		_imap = (HashMap<ConfigType, Integer>) conf._imap.clone();
	}
	
	/**
	 * 
	 * @param key
	 * @param value
	 */
	public void set( ConfigType key, boolean value ) {
		_bmap.put(key, value);
	}
	
	/**
	 * 
	 * @param key
	 * @param value
	 */
	public void set( ConfigType key, int value ) {
		_imap.put(key, value);
	}
	
	/**
	 * 
	 * @param key
	 * @return
	 */
	public boolean getBool( ConfigType key ) {
		if( _bmap.containsKey(key) )
			return _bmap.get(key);
		return false;
	}
	
	/**
	 * 
	 * @param key
	 * @return
	 */
	public int getInt( ConfigType key ) {
		if( _imap.containsKey(key) )
			return _imap.get(key);
		return -1;
	}
	
	
	/**
	 * 
	 */
	public CompilerConfig clone() {
		return new CompilerConfig(this);
	}
}
