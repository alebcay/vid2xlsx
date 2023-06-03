#!/usr/bin/env python3

import argparse
import cv2
import imutils
import numpy as np
import sys
import xlsxwriter

from imutils.video import FileVideoStream
from loguru import logger
from sklearn.cluster import MiniBatchKMeans


def process_video(input_path, output_path, colors, frame_rate, verbose=False, debug=False):
    cap = FileVideoStream(input_path).start()
    length = int(cv2.VideoCapture(input_path).get(cv2.CAP_PROP_FRAME_COUNT))

    if debug:
        logger.debug("File contains {} frames, expecting {} frames in output", length, length // frame_rate)
    
    if length // frame_rate * colors > 64000:
        logger.warning("Current settings may exceed the maximum number of colors permitted in an XLSX file (64000)")
    
    width, height = 640, 360
    count = 0
    workbook = xlsxwriter.Workbook(output_path, {'constant_memory': True})
    palette = {}
    
    while cap.more():
        frame = cap.read()
        
        if frame is None:
            break
        
        if count % frame_rate == 0:
            current = workbook.add_worksheet(str(count))
            current.set_zoom(10)
            
            if verbose:
                logger.info("Processing frame {}", count)
            
            if debug:
                logger.debug("Palette dictionary size: {}", len(palette))
            
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            (h, w) = frame.shape[:2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            frame = frame.reshape((frame.shape[0] * frame.shape[1], 3))
            
            clt = MiniBatchKMeans(n_clusters=colors)
            labels = clt.fit_predict(frame)
            quant = clt.cluster_centers_.astype("uint8")[labels]
            quant = quant.reshape((h, w, 3))
            frame = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
            
            cv2.imshow("Preview", frame)
            cv2.waitKey(1)
            
            current.set_column(0, width - 1, 3.17)
            
            for row in range(height):
                current.set_row(row, 18.75)
                
                for col in range(width):
                    blue, green, red = frame[row][col]
                    color = f'{red:02x}{green:02x}{blue:02x}'
                    
                    if color not in palette:
                        palette[color] = workbook.add_format({'bg_color': f'#{color}'})
                    
                    cell_format = palette[color]
                    current.write_blank(row, col, None, cell_format)
        
        count += 1
    
    if len(palette) > 64000:
        logger.warning("""
            Palette size ({}) exceeds the maximum permitted under the XLSX specification (64000).
            The resulting file may not be recognized as a valid XLSX file.""", len(palette))
    
    if verbose:
        logger.info("Writing XLSX file: {}", output_path)
    
    workbook.close()
    cap.stop()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input video file")
    ap.add_argument("-o", "--output", required=True, help="path to output XLSX file")
    ap.add_argument("-v", "--verbose", action="store_true", help="show detailed output")
    ap.add_argument("-d",
