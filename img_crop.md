## `img_crop`-Function

The `img_crop`-function is an image data augumentation function.
It cuts out a subregion of an image. 

### Usage
```r
img_crop(img_in = A, w = 20, h = 10, x_offset = 0, y_offset = 0)
```

### Arguments
| Name         | Type           | Default  | Description |
| :------      | :------------- | -------- | :---------- |
| img_in       | Matrix[Double] | ---      | Input matrix/image |
| w            | Integer        | ---      | The width of the subregion required  |
| h            | Integer        | ---      | The height of the subregion required |
| x_offset     | Integer        | ---      | The horizontal coordinate in the image to begin the crop operation |
| y_offset     | Integer        | ---      | The vertical coordinate in the image to begin the crop operation |

### Returns
| Name      | Type           | Default  | Description |
| :------   | :------------- | -------- | :---------- |
| img_out   | Matrix[Double] | ---      | Cropped matrix/image |

### Example
```r
A = rand (rows = 3, cols = 3, min = 0, max = 255) 
B = img_crop(img_in = A, w = 20, h = 10, x_offset = 0, y_offset = 0)
```

