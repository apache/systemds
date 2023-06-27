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

package org.apache.sysds.parser;

import java.util.HashMap;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.parser.LanguageException.LanguageErrorCodes;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

public abstract class Identifier extends Expression
{
	protected DataType _dataType;
	protected ValueType _valueType;
	protected long _dim1;
	protected long _dim2;
	protected int _blocksize;
	protected long _nnz;
	protected FileFormat _format;
	protected PrivacyConstraint _privacy;

	public Identifier() {
		_dim1 = -1;
		_dim2 = -1;
		_dataType = DataType.UNKNOWN;
		_valueType = ValueType.UNKNOWN;
		_blocksize = -1;
		_nnz = -1;
		setOutput(this);
		_format = null;
	}
	
	public void setProperties(Identifier i) {
		if (i == null) 
			return;
		_dataType = i.getDataType();
		_valueType = i.getValueType();
		if (i instanceof IndexedIdentifier) {
			_dim1 = ((IndexedIdentifier)i).getOrigDim1();
			_dim2 = ((IndexedIdentifier)i).getOrigDim2();
		}
		else {
			_dim1 = i.getDim1();
			_dim2 = i.getDim2();
		}
		_blocksize = i.getBlocksize();
		_nnz = i.getNnz();
		_format = i.getFileFormat();
		_privacy = i.getPrivacy();
	}
	
	public void setDimensionValueProperties(Identifier i) {
		if (i instanceof IndexedIdentifier) {
			IndexedIdentifier ixi = (IndexedIdentifier)i; 
			_dim1 = ixi.getOrigDim1();
			_dim2 = ixi.getOrigDim2();
		}
		else {
			_dim1 = i.getDim1();
			_dim2 = i.getDim2();
		}
		_nnz = i.getNnz();
		_dataType = i.getDataType();
		_valueType = i.getValueType();
	}
	
	public void setDataType(DataType dt){
		_dataType = dt;
	}
	
	public void setValueType(ValueType vt){
		_valueType = vt;
	}
	
	public void setFileFormat(FileFormat ft){
		_format = ft;
	}
	
	public void setDimensions(long dim1, long dim2){
		_dim1 = dim1;
		_dim2 = dim2;
	}
	
	public void setNnz(long nnzs){
		_nnz = nnzs;
	}

	public void setPrivacy(PrivacyLevel privacyLevel){
		_privacy = new PrivacyConstraint(privacyLevel);
	}

	public void setPrivacy(PrivacyConstraint privacyConstraint){
		_privacy = privacyConstraint;
	}
	
	public long getDim1(){
		return _dim1;
	}
	
	public long getDim2(){
		return _dim2;
	}
	
	public DataType getDataType(){
		return _dataType;
	}
	
	public ValueType getValueType(){
		return _valueType;
	}
	
	public FileFormat getFileFormat(){
		return _format;
	}
	
	public int getBlocksize(){
		return _blocksize;
	}
	
	public void setBlocksize(int blen){
		_blocksize = blen;
	}
	
	public long getNnz(){
		return _nnz;
	}

	public PrivacyConstraint getPrivacy(){
		return _privacy;
	}
	
	@Override
	public void validateExpression(HashMap<String,DataIdentifier> ids, HashMap<String,ConstIdentifier> constVars, boolean conditional) 
	{
		if( getOutput() instanceof DataIdentifier ) {
			// set properties for Data identifier
			String name = ((DataIdentifier)getOutput()).getName();
			Identifier id = ids.get(name);
			if ( id == null ){
				//undefined variables are always treated unconditionally as error in order to prevent common script-level bugs
				raiseValidateError("Undefined Variable (" + name + ") used in statement", false, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			getOutput().setProperties(id);
			
			// validate IndexedIdentifier -- which is substype of DataIdentifer with index
			if( getOutput() instanceof IndexedIdentifier ){
				// validate the row / col index bounds (if defined)
				IndexedIdentifier ixId = (IndexedIdentifier)getOutput();
				Expression[] exp = new Expression[]{ixId.getRowLowerBound(),
					ixId.getRowUpperBound(), ixId.getColLowerBound(), ixId.getColUpperBound()};
				String[] msg = new String[]{"row lower", "row upper", "column lower", "column upper"};
				for( int i=0; i<4; i++ ) {
					if( exp[i] != null ) {
						exp[i].validateExpression(ids, constVars, conditional);
						if (exp[i].getOutput().getDataType() == DataType.MATRIX){
							raiseValidateError("Matrix values for "+msg[i]+" index bound are "
								+ "not supported, which includes indexed identifiers.", conditional);
						}
					}
				}
				
				if( getOutput().getDataType() == DataType.LIST ) {
					int dim1 = (((IndexedIdentifier)getOutput()).getRowUpperBound() == null) ? 1 : - 1;
					((IndexedIdentifier)getOutput()).setDimensions(dim1, 1);
				} 
				else { //default
					IndexPair updatedIndices = ((IndexedIdentifier)getOutput()).calculateIndexedDimensions(ids, constVars, conditional);
					((IndexedIdentifier)getOutput()).setDimensions(updatedIndices._row, updatedIndices._col);
				}
			}
		}
		else {
			this.getOutput().setProperties(this.getOutput());
		}
	}
	
	public void computeDataType() {
		_dataType = ((_dim1 == 0) && (_dim2 == 0)) ?
			DataType.SCALAR : ((_dim1 >= 1) || (_dim2 >= 1)) ?
			DataType.MATRIX : DataType.UNKNOWN;
	}
	
	public void setBooleanProperties(){
		_dataType = DataType.SCALAR;
		_valueType = ValueType.BITSET;
		_dim1 = 0;
		_dim2 = 0;
		_blocksize = 0;
		_nnz = -1;
		_format = null;
	}
	
	public void setIntProperties(){
		_dataType = DataType.SCALAR;
		_valueType = ValueType.INT64;
		_dim1 = 0;
		_dim2 = 0;
		_blocksize = 0;
		_nnz = -1;
		_format = null;
	}
	
	
	public boolean isScalarBoolean(){
		return (_valueType == ValueType.BITSET) && (_dataType == DataType.SCALAR);
	}
	
	public boolean dimsKnown(){
		return ( _dim1 >= 0 && _dim2 >= 0);
	}
}
