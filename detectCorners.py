import cv2
import numpy as np

# Resmi oku
def detectCorners(image):
    #image = cv2.imread('cevap_anahtari.jpg')
    image = cv2.resize(image, (450, 600))
    
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Kenarları tespit et
    edges = cv2.Canny(gray, 50, 150)
    
    # Konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # En büyük konturu bul
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Köşe noktalarını tespit et
    corners = cv2.approxPolyDP(largest_contour, 0.04 * cv2.arcLength(largest_contour, True), True)
    
    # Köşe noktalarını ekrana işaretle
    for corner in corners:
        x, y = corner[0]
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    
    return corners