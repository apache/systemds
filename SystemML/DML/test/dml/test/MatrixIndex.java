package dml.test;

/**
 * <p>Index of a binary matrix cell or block.</p>
 * 
 * @author schnetter
 */
public class MatrixIndex {
    
    private int row;
    private int col;
    
    
    public MatrixIndex(int row, int col) {
        this.row = row;
        this.col = col;
    }
    
    public int getRow() {
        return row;
    }
    
    public int getCol() {
        return col;
    }
    
}
