import numpy as np
import cv2

filenames = ["Ancient Brontodon",
            "Ashes of the Abhorrent",
            "Bishop's Soldier",
            "Crash the Ramparts",
             "Deathless Ancient",
             "Headstrong Brute",
             "Jungle Delver",
             "Kitesail Freebooter",
             "March of the drowned",
             "Mountain",
             "Nest Robber",
             "One With the Wind",
             "Prying Blade",
             "Sentinel Totem",
             "Tishana's Wayfinder",
            "Vanquisher's Banner"]

def affineTran(img, pts):
    dst_pts = np.float32([[0, 0], [0, 370], [265, 370],[265, 0]])
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warp = cv2.warpPerspective(img, M, (265, 370))
    return warp

def avghash_files():
    hash_array = np.array(np.empty((370,265,len(filenames))))
    index = 0
    for file in filenames:
        img = cv2.imread("card database/" + file + '.png')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        hash_array[:,:,index] = avg_hash(gray)
        index += 1
    return hash_array

def get_filename(index):
    return filenames[index]

def avg_hash(gray):
    gray = cv2.GaussianBlur(gray, (3, 33), 0)
    avg = gray.mean()
    tmp, hash = cv2.threshold(gray, avg, 255, cv2.THRESH_BINARY)
    return hash

def hamming_dist(img1, img2):
    return sum(sum(abs(img1 - img2)))