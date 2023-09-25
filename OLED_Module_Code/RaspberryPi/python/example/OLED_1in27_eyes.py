import logging
import time
import sys
import os
picdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pic')
libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)
from waveshare_OLED import OLED_1in27_rgb
from PIL import Image, ImageDraw
import cv2
import numpy as np
import sys
import os

logging.basicConfig(level=logging.DEBUG)

def main():
    cap = cv2.VideoCapture(-1)
    
    disp = OLED_1in27_rgb.OLED_1in27_rgb()

    logging.info("1.27inch rgb OLED")
    disp.Init()
    logging.info("Clear display")
    disp.clear()

    width, height = disp.width, disp.height
    image = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    back_sub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=25, detectShadows=True)
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

        ellipse_width = 30
        ellipse_height = 20

        # Ensure the ellipse stays within display boundaries
        x2 = max(ellipse_width // 2, min(width - ellipse_width // 2, x2))
        y2 = max(ellipse_height // 2, min(height - ellipse_height // 2, y2))

        # Clear the previous frame
        draw.rectangle((0, 0, width, height), outline=0, fill=0)

        # Draw the ellipse using updated coordinates
        draw.ellipse(
            [(x2 - ellipse_width // 2, y2 - ellipse_height // 2),
             (x2 + ellipse_width // 2, y2 + ellipse_height // 2)],
            outline=(255, 0, 0), fill=(0, 0, 255))

        disp.ShowImage(disp.getbuffer(image))

        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(frame, text, (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
