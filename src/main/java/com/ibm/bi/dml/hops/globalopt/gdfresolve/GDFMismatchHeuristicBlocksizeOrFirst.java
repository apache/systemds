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

package com.ibm.bi.dml.hops.globalopt.gdfresolve;

import com.ibm.bi.dml.hops.globalopt.RewriteConfig;

public class GDFMismatchHeuristicBlocksizeOrFirst extends GDFMismatchHeuristic
{
	
	@Override
	public String getName(){
		return "BLOCKSIZE_OR_FIRST";
	}
	
	@Override
	public boolean resolveMismatch( RewriteConfig currRc, RewriteConfig newRc ) 
	{
		//check for blocksize mismatch
		if( currRc.getBlockSize() != newRc.getBlockSize() ) 
		{
			//choose the new rewrite config if its blocksize is larger than 
			//the current (intuition: we generally prefer larger blocksizes 
			//because this often enables better physical operators with constraints
			//like ncol(X) <= blocksize)
			return (currRc.getBlockSize() < newRc.getBlockSize());
		}
		
		//return the current rewrite configuration (first come first served)
		//if the previous check for blocksize mismatch failed
		return false;
	}
}
