## `img_brightness`-Function

The `img_brightness`-function is an image data augumentation function.
It changes the brightness of the image.

### Usage
```r
img_brightness(img_in = A, value = 128, channel_max = 255)
```

### Arguments
| Name         | Type           | Default  | Description |
| :------      | :------------- | -------- | :---------- |
| img_in       | Matrix[Double] | ---      | Input matrix/image |
| value        | Double         | ---      | The amount of brightness to be changed for the image |
| channel_max  | Integer        | ---      | Maximum value of the brightness of the  image |

### Returns
| Name      | Type           | Default  | Description |
| :------   | :------------- | -------- | :---------- |
| img_out   | Matrix[Double] | ---      | Output matrix/image |

### Example
```r
A = rand (rows = 3, cols = 3, min = 0, max = 255)
B = img_brightness(img_in = A, value = 128, channel_max = 255)
```

