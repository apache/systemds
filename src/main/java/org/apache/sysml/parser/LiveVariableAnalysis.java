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

package org.apache.sysml.parser;


public abstract class LiveVariableAnalysis 
{

		
	VariableSet _read;
	VariableSet _updated;
	VariableSet _gen;
	VariableSet _kill;
	VariableSet _liveIn;
	VariableSet _liveOut;
	boolean _initialized = false;
	
	VariableSet _warnSet;	// variables that may not be initialized
							// applicable for control blocks
		
	public VariableSet variablesRead(){
		return _read;
	}
	
	public VariableSet variablesUpdated() {
		return _updated;
	}
	
	public VariableSet getWarn(){
		return _warnSet;
	}
	
	public VariableSet liveIn() {
		return _liveIn;
	}
	
	public VariableSet liveOut() {
		return _liveOut;
	}
	
	public VariableSet getKill(){
		return _kill;
	}
	
	public void setLiveOut(VariableSet lo){
		_liveOut = lo;
	}
	
	public void setLiveIn(VariableSet li){
		_liveIn = li;
	}
	
	public void setUpdatedVariables( VariableSet vars ){
		_updated = vars;
	}
	
	public void setReadVariables( VariableSet vars ){
		_read = vars;
	}
	
	public abstract VariableSet initializeforwardLV(VariableSet activeIn) throws LanguageException;
	public abstract VariableSet initializebackwardLV(VariableSet loPassed) throws LanguageException;
	public abstract VariableSet analyze(VariableSet loPassed) throws LanguageException;
	
	
	public void updateLiveVariablesIn(VariableSet liveIn){
		 updateLiveVariables(_liveIn,liveIn);
	}
	
	public void updateLiveVariablesOut(VariableSet liveOut){
		 updateLiveVariables(_liveOut,liveOut);
	}
	
	private void updateLiveVariables(VariableSet origVars, VariableSet newVars){
		for (String var : newVars.getVariables().keySet()){
			if (origVars.containsVariable(var)){
				DataIdentifier varId = newVars.getVariable(var);
				if (varId != null){
					origVars.addVariable(var, varId);
				}
			}
		}
	}
}
