from imutils import paths
import argparse
import requests
import cv2
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True, help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
args = vars(ap.parse_args())

# Grab the list of URLs from the input file, then initialize the total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0

# Loop the URLs
for url in rows:
    try:
        # Try to download the image
        r = requests.get(url, timeout=60)
        # Save the image to disk
        p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(8))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()
        # Update the counter
        print("[INFO] downloaded: {}".format(p))
        total += 1
    # Handle if any exceptions are thrown during the download process
    except:
        print("[INFO] error downloading {}...skipping".format(p))

# Loop over the image paths we just downloaded
for imagePath in paths.list_images(args["output"]):
    # Initialize if the image should be deleted or not
    delete = False
    # Try to load the image
    try:
        image = cv2.imread(imagePath)
        # If the image is `None` then we could not properly load it from disk, so delete it
        if image is None:
            delete = True
    # If OpenCV cannot load the image then the image is likely corrupt so we should delete it
    except:
        print("Except")
        delete = True
    # Check to see if the image should be deleted
    if delete:
        print("[INFO] deleting {}".format(imagePath))
        os.remove(imagePath)