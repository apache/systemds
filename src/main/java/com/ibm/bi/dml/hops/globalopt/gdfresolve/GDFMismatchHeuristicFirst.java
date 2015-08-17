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

public class GDFMismatchHeuristicFirst extends GDFMismatchHeuristic
{
	
	@Override
	public String getName(){
		return "FIRST";
	}
	
	@Override
	public boolean resolveMismatch( RewriteConfig currRc, RewriteConfig newRc ) 
	{
		//always return the current rewrite configuration (first come first served)
		return false;
	}
}
