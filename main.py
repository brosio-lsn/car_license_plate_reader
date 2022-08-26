import cv2 as cv
import numpy as np
import pytesseract

# function to detect the plates
def detect_plates(img):
    # fetching the cascade files
    russian_plates_cascade = cv.CascadeClassifier("haarcascade_russian_plate_number.xml")
    # detecting the plates
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plates = russian_plates_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=3)
    # draw rectangles (to check if the detection worked fine)
    for rectangle in plates:
        x, y, w, h = rectangle.reshape(4)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return plates, img


# function that returns list with a mask for each plate
# (instead of masks, the image could be cropped, another possibility...)
def masks(plates, img):
    # initiate list that contains a mask for each plate
    masks_list = []
    for rectangle in plates:
        x, y, w, h = rectangle.reshape(4)
        # initiate a blank image with the same dimensions as the given image
        blank = np.zeros(img.shape[:2], dtype='uint8')
        # put the pixels white only if they are in the plate area
        for i in range(x, x + w):
            for j in range(y, y + h):
                blank[j][i] = 255
        masks_list.append(blank)
    return masks_list

# function that reads the text in each of the masked area in the given image
def read_text_in_each_area(masks_list, img):
    # tesseract cmd
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\ambro\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'
    # variable to count the number of plates
    count=0
    # analyse each plate
    for mask in masks_list:
        count+=1
        # mask the right area
        masked = cv.bitwise_and(img, img, mask=mask)
        # convert to grayscale for tesseract
        masked=cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
        # image transformation to improve text detection (modify the kernel size depending on the image)
        kernel = np.ones((1, 1), np.uint8)
        masked = cv.dilate(masked, kernel, iterations=1)
        masked = cv.erode(masked, kernel, iterations=1)
        # extracting the first 10 characters detected (the useful ones)
        string=pytesseract.image_to_string(masked)[0:10]
        # printing the result
        print('car number '+str(count) +":", string)

# load image
img = cv.imread('russian_car.jpg')
copy = img.copy()
# detect plates
plates, result = detect_plates(copy)
# create masks for each plate
masks_list= masks(plates, copy)
# read each plate
read_text_in_each_area(masks_list, copy)


