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

package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;

import java.util.stream.Stream;

/**
 * Used for propagating constraints in a matrix multiplication.
 */
public abstract class MatrixMultiplicationPropagator implements Propagator {

	MatrixBlock input1, input2;
	PrivacyConstraint privacyConstraint1, privacyConstraint2;

	/**
	 * Constructor for empty instance.
	 * The fields can later be set with the setFields method.
	 */
	public MatrixMultiplicationPropagator(){}

	/**
	 * Constructs the propagator and initializes the fields used for propagation.
	 * @param input1 left-hand input in matrix multiplication.
	 * @param privacyConstraint1 privacy constraint of left-hand input in matrix multiplication
	 * @param input2 right-hand input in matrix multiplication
	 * @param privacyConstraint2 privacy constraint of right-hand input in matrix multiplication
	 */
	public MatrixMultiplicationPropagator(
		MatrixBlock input1, PrivacyConstraint privacyConstraint1,
		MatrixBlock input2, PrivacyConstraint privacyConstraint2){
		setFields(input1, privacyConstraint1, input2, privacyConstraint2);
	}

	/**
	 * Sets all fields of propagator.
	 * @param input1 left-hand input in matrix multiplication.
	 * @param privacyConstraint1 privacy constraint of left-hand input in matrix multiplication
	 * @param input2 right-hand input in matrix multiplication
	 * @param privacyConstraint2 privacy constraint of right-hand input in matrix multiplication
	 */
	public void setFields(
		MatrixBlock input1, PrivacyConstraint privacyConstraint1,
		MatrixBlock input2, PrivacyConstraint privacyConstraint2){
		this.input1 = input1;
		this.privacyConstraint1 = privacyConstraint1;
		this.input2 = input2;
		this.privacyConstraint2 = privacyConstraint2;
	}

	@Override
	public PrivacyConstraint propagate() {
		// If the overall privacy level is private, then the fine-grained constraints do not have to be checked.
		if ( (privacyConstraint1 != null && privacyConstraint1.getPrivacyLevel() == PrivacyConstraint.PrivacyLevel.Private)
			|| (privacyConstraint2 != null && privacyConstraint2.getPrivacyLevel() == PrivacyConstraint.PrivacyLevel.Private) )
			return new PrivacyConstraint(PrivacyConstraint.PrivacyLevel.Private);

		int r1 = input1.getNumRows();
		int c1 = input1.getNumColumns();
		int r2 = input2.getNumRows();
		int c2 = input2.getNumColumns();
		PrivacyConstraint mergedConstraint = new PrivacyConstraint();
		FineGrainedPrivacy mergedFineGrainedConstraints = mergedConstraint.getFineGrainedPrivacy();

		// Get row privacy levels for input1
		PrivacyConstraint.PrivacyLevel[] rowPrivacy = (privacyConstraint1 != null && privacyConstraint1.getFineGrainedPrivacy() != null) ?
			privacyConstraint1.getFineGrainedPrivacy().getRowPrivacy(r1,c1) :
			Stream.generate(() -> PrivacyConstraint.PrivacyLevel.None).limit(r1).toArray(PrivacyConstraint.PrivacyLevel[]::new);
		// Get col privacy levels for input2
		PrivacyConstraint.PrivacyLevel[] colPrivacy = (privacyConstraint2 != null && privacyConstraint2.getFineGrainedPrivacy() != null) ?
			privacyConstraint2.getFineGrainedPrivacy().getColPrivacy(r2,c2) :
			Stream.generate(() -> PrivacyConstraint.PrivacyLevel.None).limit(c2).toArray(PrivacyConstraint.PrivacyLevel[]::new);
		// Get operator type array based on values in rows of input1 and cols of input2
		OperatorType[] operatorTypes1 = getOperatorTypesRow();
		OperatorType[] operatorTypes2 = getOperatorTypesCol();
		// Propagate privacy levels based on above arrays
		generateFineGrainedConstraints(mergedFineGrainedConstraints, rowPrivacy, colPrivacy, operatorTypes1, operatorTypes2);
		return mergedConstraint;
	}

	/**
	 * Gets the operator types of all rows of the left-hand input in the matrix multiplication.
	 * An operator type defines if the row will result in an aggregation or not.
	 * @return array of operator types representing the rows of the left-hand input in the matrix multiplication
	 */
	public OperatorType[] getOperatorTypesRow(){
		OperatorType[] operatorTypes = new OperatorType[input1.getNumRows()];
		for (int i = 0; i < input1.getNumRows(); i++) {
			MatrixBlock rowSlice = input1.slice(i,i);
			operatorTypes[i] = getOperatorType(rowSlice);
		}
		return operatorTypes;
	}

	/**
	 * Gets the operator types of all columns of the right-hand input in the matrix multiplication.
	 * An operator type defines if the column will result in an aggregation or not.
	 * @return array of operator types representing the columns of the right-hand input in the matrix multiplication.
	 */
	public OperatorType[] getOperatorTypesCol(){
		OperatorType[] operatorTypes = new OperatorType[input2.getNumColumns()];
		for (int j = 0; j < input2.getNumColumns(); j++) {
			MatrixBlock colSlice = input2.slice(0, input2.getNumRows()-1, j, j, new MatrixBlock());
			operatorTypes[j] = getOperatorType(colSlice);
		}
		return operatorTypes;
	}

	protected OperatorType mergeOperatorType(OperatorType input1, OperatorType input2){
		if (input1 == OperatorType.NonAggregate || input2 == OperatorType.NonAggregate)
			return OperatorType.NonAggregate;
		else return OperatorType.Aggregate;
	}

	protected OperatorType getOperatorType(MatrixBlock inputSlice){
		if(inputSlice.getNonZeros() == 1)
			return OperatorType.NonAggregate;
		else
			return OperatorType.Aggregate;
	}

	/**
	 * An abstract method that generates the fine-grained privacy constraints based on the input
	 * and puts it in the mergedFineGrainedConstraints.
	 * @param mergedFineGrainedConstraints FineGrainedPrivacy instance in which the output privacy constraints are put
	 * @param rowPrivacy privacy constraints of the rows of the left-hand input in the matrix multiplication
	 * @param colPrivacy privacy constraints of the columns of the right-hand input in the matrix multiplication
	 * @param operatorTypes1 operator types of the left-hand input in the matrix multiplication
	 * @param operatorTypes2 operator types of the right-hand input in the matrix multiplication
	 */
	protected abstract void generateFineGrainedConstraints(FineGrainedPrivacy mergedFineGrainedConstraints,
		PrivacyConstraint.PrivacyLevel[] rowPrivacy, PrivacyConstraint.PrivacyLevel[] colPrivacy,
		OperatorType[] operatorTypes1, OperatorType[] operatorTypes2);

}
