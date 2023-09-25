import logging
import time
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

    # Define eye parameters
    eye_radius = 30
    eye_distance = 40

    # Initialize previous eye positions
    prev_left_eye_x = width // 2 - eye_distance
    prev_right_eye_x = width // 2 + eye_distance
    prev_eye_y = height // 2

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

        ellipse_width = 40
        ellipse_height = 80

        # Ensure the ellipses stay within display boundaries
        x2 = max(ellipse_width // 2, min(width - ellipse_width // 2, x2))
        y2 = max(ellipse_height // 2, min(height - ellipse_height // 2, y2))

        # Interpolate eye positions for smooth transition
        alpha = 0.2  # Smoothing factor
        left_eye_x = int((1 - alpha) * prev_left_eye_x + alpha * (x2 - eye_distance))
        right_eye_x = int((1 - alpha) * prev_right_eye_x + alpha * (x2 + eye_distance))
        eye_y = int((1 - alpha) * prev_eye_y + alpha * y2)

        # Ensure eyes stay within display boundaries
        left_eye_x = max(eye_radius, min(width - eye_radius, left_eye_x))
        right_eye_x = max(eye_radius, min(width - eye_radius, right_eye_x))
        eye_y = max(eye_radius, min(height - eye_radius, eye_y))

        prev_left_eye_x = left_eye_x
        prev_right_eye_x = right_eye_x
        prev_eye_y = eye_y

        # Clear the previous frame
        draw.rectangle((0, 0, width, height), outline=0, fill=0)

        # Draw two ellipses to represent eyes
        draw.ellipse(
            [(left_eye_x - eye_radius, eye_y - eye_radius),
             (left_eye_x + eye_radius, eye_y + eye_radius)],
            outline=(255, 255, 255), fill=(255, 255, 255))

        draw.ellipse(
            [(right_eye_x - eye_radius, eye_y - eye_radius),
             (right_eye_x + eye_radius, eye_y + eye_radius)],
            outline=(255, 255, 255), fill=(255, 255, 255))

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
