import cv2
import numpy as np
import util

feature_params = dict(  maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

# Calculate the hashes of card files
hash_files = util.avghash_files()

# Setup the camera
camera = cv2.VideoCapture(0)

# Read and display input image
_, img = camera.read()

# Loop Forever
while True:
    # Loop and display webcam images until enter key is pressed
    while True:
        _, img = camera.read()
        cv2.imshow('Input Image',img)
        if cv2.waitKey(0) & 0xff == 13:
            break

    # Convert the image to grayscale, smooth it, and display it
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    cv2.imshow("Grayscale Input", gray)

    # Edge detection on the grayscale image
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    cv2.imshow("edges", edges)

    # Line detection
    line_img = np.zeros(gray.shape)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=50, minLineLength=20, maxLineGap=300)

    # Draw the lines on a new image, this helps get stronger features
    size = lines.shape[0]
    for i in range(0,size):
        x1, y1, x2, y2 = lines[i,0,:]
        cv2.line(line_img,(x1,y1),(x2,y2),255,2)
    cv2.imshow("lines", line_img)
    line_img = np.float32(line_img)

    # Find corners in the lines only image
    features = cv2.goodFeaturesToTrack(line_img, mask = None, **feature_params)
    features = features.reshape(features.shape[0],2)

    # Find the highest, lowest, rightmost, and left most features, we assume these are the 4 card corners
    arg = np.argsort(features, axis=0)
    left = features[arg[0][0]]
    right = features[arg[-1][0]]
    bot = features[arg[-1][1]]
    top = features[arg[0][1]]

    # Sort the corners ordered: top-left, bot-left, top-right, bot-right
    # We assume the card is upright so it can either be slanted right or left
    if(left[1] < right[1]):
        card_points = np.array([left,bot,right,top])
    else:
        card_points = np.array([top,left,bot,right])

    # Display located features and the corners
    circle_features = tuple(map(tuple,features))
    feature_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    circle_corners  = tuple(map(tuple,card_points))
    for center in circle_corners:
        cv2.circle(feature_img,center, 5, (255,0,0), -1)
    for center in circle_features:
        cv2.circle(feature_img,center, 2, (0,0,255), -1)
    cv2.imshow("Features Located",feature_img)

    # Transform the card to the standard size
    card = util.affineTran(gray, card_points)
    cv2.imshow('Transformed Card',card)

    # Calculate the average hash
    hash = util.avg_hash(card)

    # Find the card with the minimum hamming distance from our image card
    mini = float('Inf')
    minVal=0
    for i in range(0,hash_files.shape[2]):
        dist = util.hamming_dist(hash_files[:,:,i],hash)
        if dist < mini:
            mini = dist
            minVal = i
    print(util.get_filename(minVal))


    # End program if key is pressed
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
        exit()
