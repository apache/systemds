package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;

import java.util.stream.Stream;

public abstract class MatrixMultiplicationPropagator implements Propagator {

	MatrixBlock input1, input2;
	PrivacyConstraint privacyConstraint1, privacyConstraint2;

	public MatrixMultiplicationPropagator(){};

	public MatrixMultiplicationPropagator(
		MatrixBlock input1, PrivacyConstraint privacyConstraint1,
		MatrixBlock input2, PrivacyConstraint privacyConstraint2){
		setFields(input1, privacyConstraint1, input2, privacyConstraint2);
	}

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

	public OperatorType[] getOperatorTypesRow(){
		OperatorType[] operatorTypes = new OperatorType[input1.getNumRows()];
		for (int i = 0; i < input1.getNumRows(); i++) {
			MatrixBlock rowSlice = input1.slice(i,i);
			operatorTypes[i] = getOperatorType(rowSlice);
		}
		return operatorTypes;
	}

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

	protected abstract void generateFineGrainedConstraints(FineGrainedPrivacy mergedFineGrainedConstraints,
		PrivacyConstraint.PrivacyLevel[] rowPrivacy, PrivacyConstraint.PrivacyLevel[] colPrivacy,
		OperatorType[] operatorTypes1, OperatorType[] operatorTypes2);

}
