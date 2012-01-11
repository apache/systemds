package dml.test;


/**
 * <p>Contains characteristics about a binary matrix.</p>
 * 
 * @author schnetter
 */
public class BinaryMatrixCharacteristics {
    
    private double[][] values;
    private int rows;
    private int cols;
    private int rowsInBlock;
    private int rowsInLastBlock;
    private int colsInBlock;
    private int colsInLastBlock;
    private long nonZeros;
    
    
    public BinaryMatrixCharacteristics(double[][] values, int rows, int cols, int rowsInBlock, int rowsInLastBlock,
            int colsInBlock, int colsInLastBlock, long nonZeros) {
        this.values = values;
        this.rows = rows;
        this.cols = cols;
        this.rowsInBlock = rowsInBlock;
        this.rowsInLastBlock = rowsInLastBlock;
        this.colsInBlock = colsInBlock;
        this.colsInLastBlock = colsInLastBlock;
        this.nonZeros = nonZeros;
    }
    
    public double[][] getValues() {
        return values;
    }
    
    public int getRows() {
        return rows;
    }
    
    public int getCols() {
        return cols;
    }
    
    public int getRowsInBlock() {
        return rowsInBlock;
    }
    
    public int getRowsInLastBlock() {
        return rowsInLastBlock;
    }
    
    public int getColsInBlock() {
        return colsInBlock;
    }
    
    public int getColsInLastBlock() {
        return colsInLastBlock;
    }
    
    public long getNonZeros() {
    	return nonZeros;
    }
    
}
