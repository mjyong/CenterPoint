# One predicted mode of an Object: positions at Object.prediction_times in the ObjectArray frame.
float32 probability
geometry_msgs/Point[] positions    # z = 0
float64[] covariance               # 2 x 2 of [x, y] per position, row major (4 values each)
