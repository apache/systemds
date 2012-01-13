package dml.sql.sqlcontrolprogram;

import java.util.ArrayList;

public interface SQLBlockContainer extends ISQLBlock {
	ArrayList<ISQLBlock> get_blocks();
}
