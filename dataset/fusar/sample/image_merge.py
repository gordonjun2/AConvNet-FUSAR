import cv2
import matplotlib.pyplot as plt
import numpy as np

first_target = cv2.imread('./Fishing/fishing_1.tiff', cv2.IMREAD_GRAYSCALE)
second_target = cv2.imread('./BulkCarrier/bulkcarrier_1.tiff', cv2.IMREAD_GRAYSCALE)

combined_target = np.zeros((first_target.shape[0], first_target.shape[1]))

power_factor = 0.75                                         # from 0 to 1

for i in range(combined_target.shape[0]):
    for j in range(combined_target.shape[1]):
        if first_target[i][j] >= second_target[i][j]:
            combined_target[i][j] = first_target[i][j]
        else:
            combined_target[i][j] = second_target[i][j] * power_factor

combined_target = np.round(combined_target)
combined_target = np.expand_dims(combined_target, axis=2)

cv2.imwrite('bulkcarrier_and_fishing_combined_decreased_power_0.75_pattern_1.jpg', combined_target)