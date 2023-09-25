import logging
import time
from waveshare_OLED import OLED_1in27_rgb
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.DEBUG)

def main():
    disp = OLED_1in27_rgb.OLED_1in27_rgb()

    logging.info("1.27inch rgb OLED")
    # Initialize library.
    disp.Init()
    # Clear display.
    logging.info("Clear display")
    disp.Clear()

    # Create blank image for drawing.
    width, height = disp.width, disp.height
    image = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    while True:
        # Replace this with your logic to update ellipse coordinates
        x2, y2 = 64, 64  # Example coordinates

        # Ensure the ellipse stays within display boundaries
        ellipse_width = 30
        ellipse_height = 20
        x2 = max(ellipse_width // 2, min(width - ellipse_width // 2, x2))
        y2 = max(ellipse_height // 2, min(height - ellipse_height // 2, y2))

        # Clear the previous frame
        draw.rectangle((0, 0, width, height), outline=0, fill=0)

        # Draw the ellipse using updated coordinates
        draw.ellipse(
            [(x2 - ellipse_width // 2, y2 - ellipse_height // 2),
             (x2 + ellipse_width // 2, y2 + ellipse_height // 2)],
            outline=(255, 0, 0), fill=(0, 0, 255))  # Example colors

        # Display the image on the OLED screen
        disp.ShowImage(disp.getbuffer(image))
 
        # Sleep to control the frame rate
        time.sleep(0.1)

if __name__ == '__main__':
    main()
