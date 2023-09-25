import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1351
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import sys
import os
import logging
import time
import traceback
from waveshare_OLED import OLED_1in27_rgb

# Raspberry Pi hardware SPI configuration for the OLED display.
RST = 24
DC = 23
SPI_PORT = 0
SPI_DEVICE = 0

# Initialize the display
disp = Adafruit_SSD1351.SSD1351_128_128(rst=RST, dc=DC, spi=SPI.SpiDev(SPI_PORT, SPI_DEVICE, max_speed_hz=8000000))

# Create a blank image with mode 'RGB'
width, height = disp.width, disp.height
image = Image.new('RGB', (width, height))

# Create a draw object
draw = ImageDraw.Draw(image)

def main():
    # Create a VideoCapture object
    cap = cv2.VideoCapture(-2)

    # Create the background subtractor object
    back_sub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=25, detectShadows=True)

    # Create kernel for morphological operation
    kernel = np.ones((20, 20), np.uint8)

    while True:
        ret, frame = cap.read()

        fg_mask = back_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        fg_mask_bb = fg_mask
        contours, hierarchy = cv2.findContours(fg_mask_bb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]

        if len(areas) < 1:
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue
        else:
            max_index = np.argmax(areas)

        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        x2 = x + int(w / 2)
        y2 = y + int(h / 2)

        # Clamp ellipse coordinates within display boundaries
        x2 = max(ellipse_width // 2, min(width - ellipse_width // 2, x2))
        y2 = max(ellipse_height // 2, min(height - ellipse_height // 2, y2))

        # Clear the previous frame
        draw.rectangle((0, 0, width, height), outline=0, fill=0)

        # Draw the ellipse using updated coordinates
        draw.ellipse(
            [(x2 - ellipse_width // 2, y2 - ellipse_height // 2),
             (x2 + ellipse_width // 2, y2 + ellipse_height // 2)],
            outline=ellipse_color, fill=None)

        # Display the image on the OLED screen
        disp.image(image)
        disp.display()

        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(frame, text, (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    main()
