import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Function for run-length decoding
def rl_decode(enc):
    # Constants for image dimensions
    HEIGHT = 1400
    WIDTH = 875

    parts = [int(s) for s in enc.split(' ')]
    dec = list()
    for i in range(0, len(parts), 2):
        cnt = parts[i]
        val = parts[i + 1]
        dec += cnt * [val]
    return np.array(dec, dtype=np.uint8).reshape((HEIGHT, WIDTH))


def rl_encode(mask):
    pixels = mask.flatten()
    rle = []
    last_value = pixels[0]
    count = 1
    for i, pixel in enumerate(pixels[1:]):
        if pixel != last_value:
            rle.append(count)
            rle.append(last_value)

            last_value = pixel
            count = 1
        else:
            count += 1

    rle.append(count)
    rle.append(last_value)


    return ' '.join(str(x) for x in rle)



# # Test data
# mask = np.zeros((1400, 875), dtype=np.uint8)
# mask[10:50, 100:200] = 1  # Create a small rectangular region of 1's
# mask[100:150, 300:350] = 1  # Create another small rectangular region of 1's
#
# # Perform run-length encoding
# encoded = rl_encode(mask)
# print(f"Encoded RLE: {encoded}")
#
# # Perform run-length decoding
# decoded_mask = rl_decode(encoded)
#
# # Check if decoded mask matches the original mask
# print(f"Masks are identical: {np.array_equal(mask, decoded_mask)}")
