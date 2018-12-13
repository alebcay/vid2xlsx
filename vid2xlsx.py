#!/usr/bin/env python3

from imutils.video import FileVideoStream
from loguru import logger
from sklearn.cluster import MiniBatchKMeans

import argparse
import cv2
import imutils
import numpy as np
import sys
import xlsxwriter

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "path to input video file")
ap.add_argument("-o", "--output", required = True, help = "path to output xlsx file")
ap.add_argument("-v", "--verbose", action = "store_true", help = "show detailed output")
ap.add_argument("-d", "--debug", action = "store_true", help = "show debug and troubleshooting information")
ap.add_argument("-c", "--colors", required = True, type = int, help = "number of colors per frame")
ap.add_argument("-f", "--frame", required = True, type = int, help = "how often a frame should be generated")
args = vars(ap.parse_args())

cap = FileVideoStream(args["input"]).start()
if args["verbose"]: logger.info("Opened FileVideoStream from {}", args["input"])
length = int(cv2.VideoCapture(args["input"]).get(7))
if args["debug"]: logger.debug("File contains {} frames, expect {} frames in output", length, length // args["frame"])
if args["debug"]: logger.debug("Projected maximum color usage: {}", length // args["frame"] * args["colors"])
if length // args["frame"] * args["colors"] > 64000: logger.warning("Current settings may exceed the maximum number of colors permitted in an XLSX file (64000)")
width = 640
height = 360
count = 0

workbook = xlsxwriter.Workbook(args["output"], {'constant_memory': True})

palette = {}
clt = None
labels = None
quant = None

while cap.more():
	frame = cap.read()
	if frame is None:
		break
	if count % args["frame"] == 0:
		current = workbook.add_worksheet(str(count))
		current.set_zoom(10)
		if args["verbose"]: logger.info("Processing frame {}", count)
		if args["debug"]: logger.debug("Palette dictionary size: {}", len(palette))
		frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_CUBIC)
		(h, w) = frame.shape[:2]
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
		frame = frame.reshape((frame.shape[0] * frame.shape[1], 3))
		clt = MiniBatchKMeans(n_clusters = args["colors"])
		labels = clt.fit_predict(frame)
		quant = clt.cluster_centers_.astype("uint8")[labels]
		quant = quant.reshape((h, w, 3))
		frame = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
		cv2.imshow("Preview", frame)
		cv2.waitKey(1)
		current.set_column(0, width - 1, 3.17)
		for row in range(0, height):
			current.set_row(row, 18.75)
			for col in range(0, width):
				blue, green, red = frame[row][col]
				color = f'{red:02x}{green:02x}{blue:02x}'
				if color not in palette:
					palette[color] = workbook.add_format({'bg_color': f'#{color}'})
				cell_format = palette[color]
				current.write_blank(row, col, None, cell_format)
	count = count + 1

if len(palette) > 64000: logger.warning("""
	Palette size ({}) exceeds the maximum permitted under the XLSX specification (64000).
	The resulting file may not be recognized as a valid XLSX file.""", len(palette))
if args["verbose"]: logger.info("Writing xlsx file: {}", args["output"])
workbook.close()
cap.stop()
cv2.destroyAllWindows()
