import cv2
import numpy as np
from opticMatching import matching
from perspectiveTransform import applyPerspectiveTransform

def process_image(image):
    kernel = np.ones((5,5),np.uint8)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to create a binary image
    #_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _,thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    
    erosion = cv2.erode(thresh, kernel, iterations = 3)
    dilation = cv2.dilate(erosion, kernel, iterations = 3)
    #opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    return dilation

def find_contours(image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def detect_bubbles(image1, image2):
    # Read the images
    #image1 = cv2.imread(image_path1)
    #image2 = cv2.imread(image_path2)

    # Preprocess the first image
    processed_image1 = process_image(np.copy(image1))
    cv2.imshow("processed_image1", processed_image1)
    contours1 = find_contours(processed_image1)

    # Preprocess the second image
    processed_image2 = process_image(np.copy(image2))
    cv2.imshow("processed_image2", processed_image2)
    contours2 = find_contours(processed_image2)

    # Count and draw green rectangles for each contour in the first image
    count_red_rectangles = 0
    for contour1 in contours1:
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        if cv2.contourArea(contour1) > 50:
            is_intersecting = False
            # Iterate through each contour in the second image
            for contour2 in contours2:
                x2, y2, w2, h2 = cv2.boundingRect(contour2)
                if cv2.contourArea(contour2) > 50:
                    # Check if contours intersect
                    if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
                        is_intersecting = True
                        break
            
            # Draw rectangles based on intersection
            if is_intersecting:
                cv2.rectangle(image1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)  # Red for intersecting
                count_red_rectangles += 1
            else:
                cv2.rectangle(image1, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)  # Green for non-intersecting

    # Display the result with the count in the top-left corner
    dogru = 20 - count_red_rectangles
    puan = dogru * 5
    cv2.putText(image1, f'dogru sayisi: {dogru}, puan: {puan}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("soru", image1)
    cv2.imshow("cevap", image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture('http://192.168.5.197:8080/video')
    
    image_path1 = cv2.imread("optik_yeni.png")
    image_path2 = cv2.imread("optik_yeni2.png")
    cevap_anahtari = cv2.imread("cevap_anahtari.jpg")
    cevap_anahtari = cv2.resize(cevap_anahtari, (450, 600))
    transformed_cevap_anahtari = applyPerspectiveTransform(cevap_anahtari)
        
    new_width = 700
    new_height = 450
 
    while True:
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frame = cv2.rotate(resized_frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Replace these paths with the paths to your two bubble sheet images
        deneme = matching(resized_frame, cevap_anahtari)
        #cv2.imshow('resized_frame', resized_frame)
        cv2.imshow('cevap_anahtari', cevap_anahtari)
        
        if deneme is not None:
            #rotated_image = cv2.rotate(deneme, cv2.ROTATE_90_CLOCKWISE)
            try:
                transformed_image = applyPerspectiveTransform(deneme)
                detect_bubbles(transformed_image, transformed_cevap_anahtari)
            except:
                pass
        cv2.imshow('frame',resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break