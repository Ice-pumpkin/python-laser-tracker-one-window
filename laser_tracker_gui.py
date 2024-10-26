#! /usr/bin/env python
import numpy
import sys
import cv2
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image
from PIL import ImageTk
import cvzone

# initialize
"""
* ``cam_width`` x ``cam_height`` -- This should be the size of the
image coming from the camera. Default is 640x480.

HSV color space Threshold values for a RED laser pointer are determined
by:

* ``hue_min``, ``hue_max`` -- Min/Max allowed Hue values
* ``sat_min``, ``sat_max`` -- Min/Max allowed Saturation values
* ``val_min``, ``val_max`` -- Min/Max allowed pixel values

If the dot from the laser pointer doesn't fall within these values, it
will be ignored.

* ``display_thresholds`` -- if True, additional windows will display
  values for threshold image channels.

"""
cam_width = 640
cam_height = 480
hue_min = 20
hue_max = 160
sat_min = 100
sat_max = 255
val_min = 200
val_max = 256
display_thresholds = FALSE
capture = None  # camera capture device
channels = {
    'hue': None,
    'saturation': None,
    'value': None,
    'laser': None,
}

previous_position = None
trail = numpy.zeros((cam_height, cam_width, 3),
                    numpy.uint8)

# setup_camera_capture
"""Perform camera setup for the device number (default device = 0).
Returns a reference to the camera Capture object.

"""
try:
    device = int(0)
    sys.stdout.write("Using OpenCV version: {0}\n".format(cv2.__version__))
    sys.stdout.write("Using Camera Device: {0}\n".format(device))
except (IndexError, ValueError):
    # assume we want the 1st device
    device = 0
    sys.stderr.write("Invalid Device. Using default device 0\n")

# Try to start capturing frames
capture = cv2.VideoCapture(device)
if not capture.isOpened():
    sys.stderr.write("Faled to Open Capture device. Quitting.\n")
    sys.exit(1)

# set the wanted image size from the camera
capture.set(
    cv2.cv.CV_CAP_PROP_FRAME_WIDTH if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_WIDTH,
    cam_width
)
capture.set(
    cv2.cv.CV_CAP_PROP_FRAME_HEIGHT if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_HEIGHT,
    cam_height
)


# method that threshold image
def threshold_image(channel):
    global maximum, minimum
    if channel == "hue":
        minimum = hue_min
        maximum = hue_max
    elif channel == "saturation":
        minimum = sat_min
        maximum = sat_max
    elif channel == "value":
        minimum = val_min
        maximum = val_max

    (t, tmp) = cv2.threshold(
        channels[channel],  # src
        maximum,  # threshold value
        0,  # we dont care because of the selected type
        cv2.THRESH_TOZERO_INV  # t type
    )

    (t, channels[channel]) = cv2.threshold(
        tmp,  # src
        minimum,  # threshold value
        255,  # maxvalue
        cv2.THRESH_BINARY  # type
    )

    if channel == 'hue':
        # only works for filtering red color because the range for the hue
        # is split
        channels['hue'] = cv2.bitwise_not(channels['hue'])


# submethod of detect, track laser pointer, return detected image
def track(frame, mask):
    """
    Track the position of the laser pointer.

    Code initially taken from
    https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
    """
    center = None
    countours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[-2]

    # only proceed if at least one contour was found
    if len(countours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(countours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        moments = cv2.moments(c)
        if moments["m00"] > 0:
            center = int(moments["m10"] / moments["m00"]), \
                int(moments["m01"] / moments["m00"])
        else:
            center = int(x), int(y)

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            bbox = (int(x) - int(radius), int(y) - int(radius), int(radius) * 2, int(radius) * 2)
            frame = cvzone.cornerRect(frame, bbox, l=15, t=2, rt=0, colorC=(193, 196, 98))
    return frame


# method that detect laser pointer and return images
def detect(frame):
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # split the video frame into color channels
    h, s, v = cv2.split(hsv_img)
    channels['hue'] = h
    channels['saturation'] = s
    channels['value'] = v

    # Threshold ranges of HSV components; storing the results in place
    threshold_image("hue")
    threshold_image("saturation")
    threshold_image("value")

    # Perform an AND on HSV components to identify the laser!
    channels['laser'] = cv2.bitwise_and(
        channels['hue'],
        channels['value']
    )
    channels['laser'] = cv2.bitwise_and(
        channels['saturation'],
        channels['laser']
    )
    l = channels['laser']
    # Merge the HSV components back together.
    hsv_image = cv2.merge([
        channels['hue'],
        channels['saturation'],
        channels['value'],
    ])

    track(frame, channels['laser'])

    # return hsv_image
    return frame, hsv_image, h, s, v, l


# Set up GUIs
window = ttk.Window("Laser-Pointer-Identifier", themename="litera", size=(850, 700))
# Set up Frame
titleFrame = ttk.Frame(window, style=f'{DISABLED}')
titleFrame.grid(padx=10, pady=0, ipadx=0, ipady=0, row=0, column=0)

CameraFrame = ttk.LabelFrame(titleFrame, text="Camera", style=f'{INFO}')
CameraFrame.grid(padx=0, pady=5, ipadx=0, ipady=2, row=1, column=0)

SouthFrame = ttk.Frame(window, style=f'{DISABLED}')
SouthFrame.grid(padx=0, pady=6, ipadx=0, ipady=2, row=1, column=0)

HueFrame = ttk.LabelFrame(SouthFrame, text="Hue", style=f'{INFO}')
HueFrame.grid(padx=10, pady=0, ipadx=0, ipady=2, row=0, column=0, sticky=tk.SW)

SaturationFrame = ttk.LabelFrame(SouthFrame, text="Saturation", style=f'{INFO}')
SaturationFrame.grid(padx=10, pady=0, ipadx=0, ipady=2, row=0, column=1, sticky=tk.EW)

EastFrame = ttk.Frame(window, style=f'{DISABLED}')
EastFrame.grid(padx=6, pady=0, ipadx=0, ipady=0, row=0, column=1, sticky=tk.NE)

HSVFrame = ttk.LabelFrame(EastFrame, text="HSV", style=f'{INFO}')
HSVFrame.grid(padx=0, pady=10, ipadx=0, ipady=2, row=1, column=0, sticky=tk.NE)

ValueFrame = ttk.LabelFrame(EastFrame, text="Value", style=f'{INFO}')
ValueFrame.grid(padx=0, pady=0, ipadx=0, ipady=2, row=2, column=0, sticky=tk.NE)

LaserFrame = ttk.LabelFrame(window, text="Laser", style=f'{INFO}')
LaserFrame.grid(padx=6, pady=6, ipadx=0, ipady=2, row=1, column=1, sticky=tk.SE)

# Set up Labels that contain image stream:
TitleLabel = ttk.Label(titleFrame, text="Laser-Pointer-Identifier", style=f'{PRIMARY}')
TitleLabel.grid(padx=0, pady=5, row=0, column=0, sticky=tk.SW)
TitleLabel.configure(font=("Segoe UI", 15))

camera_label = ttk.Label(CameraFrame)
camera_label.grid(padx=6, pady=0, row=1, column=0)

blankLabel = ttk.Label(EastFrame, text="Real-time Face", foreground="white")
blankLabel.grid(padx=0, pady=5, row=0, column=0, sticky=tk.NW)
blankLabel.configure(font=("Segoe UI", 23))

HSV_label = ttk.Label(HSVFrame)
HSV_label.grid(padx=6, pady=0, row=0, column=0)

Hue_label = ttk.Label(HueFrame)
Hue_label.grid(padx=6, pady=0, row=0, column=0)

Saturation_label = ttk.Label(SaturationFrame)
Saturation_label.grid(padx=6, pady=0, row=0, column=0)

Value_label = ttk.Label(ValueFrame)
Value_label.grid(padx=6, pady=0, row=0, column=0)

Laser_label = ttk.Label(LaserFrame)
Laser_label.grid(padx=6, pady=0, row=0, column=0)


# method that capture img, run detector, and refresh GUI
def show_frame():
    _, img = capture.read()
    img = cv2.flip(img, 1)
    main_img, hsv_img, h_img, s_img, v_img, l_img = detect(img)

    resizevalue = 0.3
    main_img = cv2.resize(main_img, (0, 0), None, 0.8, 0.8)
    hsv_img = cv2.resize(hsv_img, (0, 0), None, resizevalue, resizevalue)
    h_img = cv2.resize(h_img, (0, 0), None, resizevalue, resizevalue)
    s_img = cv2.resize(s_img, (0, 0), None, resizevalue, resizevalue)
    v_img = cv2.resize(v_img, (0, 0), None, resizevalue, resizevalue)
    l_img = cv2.resize(l_img, (0, 0), None, resizevalue, resizevalue)

    main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    main_img = Image.fromarray(main_img)
    main_img_tk = ImageTk.PhotoImage(image=main_img)
    camera_label.imgtk = main_img_tk
    camera_label.configure(image=main_img_tk)
    camera_label.after(10, show_frame)

    hsv_img = Image.fromarray(hsv_img)
    hsv_img_tk = ImageTk.PhotoImage(image=hsv_img)
    HSV_label.imgtk = hsv_img_tk
    HSV_label.configure(image=hsv_img_tk)

    h_img = Image.fromarray(h_img)
    h_img_tk = ImageTk.PhotoImage(image=h_img)
    Hue_label.imgtk = h_img_tk
    Hue_label.configure(image=h_img_tk)

    s_img = Image.fromarray(s_img)
    s_img_tk = ImageTk.PhotoImage(image=s_img)
    Saturation_label.imgtk = s_img_tk
    Saturation_label.configure(image=s_img_tk)

    v_img = Image.fromarray(v_img)
    v_img_tk = ImageTk.PhotoImage(image=v_img)
    Value_label.imgtk = v_img_tk
    Value_label.configure(image=v_img_tk)

    l_img = Image.fromarray(l_img)
    l_img_tk = ImageTk.PhotoImage(image=l_img)
    Laser_label.imgtk = l_img_tk
    Laser_label.configure(image=l_img_tk)


show_frame()
window.mainloop()  # Starts GUI
