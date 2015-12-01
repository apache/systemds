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

public class ProgramRewriteStatus 
{
	
	//status of applied rewrites
	private boolean _rmBranches = false; //removed branches
	private int _blkSize = -1;
	private boolean _injectCheckpoints = false;
	
	//current context
	private boolean _inParforCtx = false;
	
	public ProgramRewriteStatus()
	{
		_rmBranches = false;
		_inParforCtx = false;
		_injectCheckpoints = false;
	}
	
	public void setRemovedBranches(){
		_rmBranches = true;
	}
	
	public boolean getRemovedBranches(){
		return _rmBranches;
	}
	
	public void setInParforContext(boolean flag){
		_inParforCtx = flag;
	}
	
	public boolean isInParforContext(){
		return _inParforCtx;
	}
	
	public void setBlocksize( int blkSize ){
		_blkSize = blkSize;
	}
	
	public int getBlocksize() {
		return _blkSize;
	}
	
	public void setInjectedCheckpoints(){
		_injectCheckpoints = true;
	}
	
	public boolean getInjectedCheckpoints(){
		return _injectCheckpoints;
	}
}
