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

package com.ibm.bi.dml.runtime.transform;


public class TfAgents {
	
	
	OmitAgent _oa = null;
	MVImputeAgent _mia = null;
	RecodeAgent _ra = null;	
	BinAgent _ba = null;
	DummycodeAgent _da = null;
	
	public TfAgents(OmitAgent oa, MVImputeAgent mia, RecodeAgent ra, BinAgent ba, DummycodeAgent da)  {
		_oa = oa;
		_mia = mia;
		_ra = ra;
		_ba = ba;
		_da = da;
	}
	
	public OmitAgent 	  getOmitAgent() 	{ 	return _oa; }
	public MVImputeAgent  getMVImputeAgent(){ 	return _mia;}
	public RecodeAgent 	  getRecodeAgent() 	{ 	return _ra; }
	public BinAgent 	  getBinAgent() 	{ 	return _ba; }
	public DummycodeAgent getDummycodeAgent() { return _da; }
	
}
