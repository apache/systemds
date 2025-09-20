package org.apache.sysds.runtime.einsum;

public class EOpNodeBinary extends EOpNode {
    public enum EBinaryOperand { // upper case: char has to remain, lower case: to be summed
        ////// summations:   //////
        aB_a,// -> B
        Ba_a, // -> B
        Ba_aC, // mmult -> BC
        aB_Ca,
        Ba_Ca, // -> BC
        aB_aC, // outer mult, possibly with transposing first -> BC
        a_a,// dot ->

        ////// elementwisemult and sums, something like ij,ij->i   //////
        aB_aB,// elemwise and colsum -> B
        Ba_Ba, // elemwise and rowsum ->B
        Ba_aB, // elemwise, either colsum or rowsum -> B
		aB_Ba,

        ////// elementwise, no summations:   //////
        A_A,// v-elemwise -> A
        AB_AB,// M-M elemwise -> AB
        AB_BA, // M-M.T elemwise -> AB
        AB_A, // M-v colwise -> BA!?
        BA_A, // M-v rowwise -> BA
        ab_ab,//M-M sum all
        ab_ba, //M-M.T sum all
        ////// other   //////
        A_B, // outer mult -> AB
        A_scalar, // v-scalar
        AB_scalar, // m-scalar
        scalar_scalar
    }
    public EOpNode left;
    public EOpNode right;
    public EBinaryOperand operand;
    public EOpNodeBinary(Character c1, Character c2, EOpNode left, EOpNode right, EBinaryOperand operand){
        super(c1,c2);
        this.left = left;
        this.right = right;
        this.operand = operand;
    }
}
