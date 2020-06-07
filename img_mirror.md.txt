## `img_mirror`-Function

The `img_mirror`-function is an image data augumentation function.
It flips an image on the x (horizontal) or y (vertical) axis. 

### Usage
```r
img_mirror(img_in = A, horizontal_axis = TRUE)
```

### Arguments
| Name              | Type           | Default  | Description |
| :------           | :------------- | -------- | :---------- |
| img_in            | Matrix[Double] | ---      | Input matrix/image |
| horizontal_axis   | Boolean        | ---      | If TRUE, the  image is flipped with respect to horizontal axis otherwise vertical axis |

### Returns
| Name      | Type           | Default  | Description |
| :------   | :------------- | -------- | :---------- |
| img_out   | Matrix[Double] | ---      | Flipped matrix/image |

### Example
```r
A = rand (rows = 3, cols = 3, min = 0, max = 255)
B = img_mirror(img_in = A, horizontal_axis = TRUE)
```

