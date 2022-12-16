import sys, os
import PIL
from PIL import Image
import random

#
#
#   Ovaj program ce se samo jednom pokrenuti!! 
#   To je samo da izgenerira dataset (nije preprocess)
#       sada preprocess moze biti samo za ono sto je bitno, npr pretvaranje u tensor
#
#
#


originalImagesFolder = "resources\original_images\\" 
# tu se moze staviti i absolute path da vuce iz vaseg diska ili cega vec
# ja stavio tih par slika tek tolko da isprobavam kako izgledaju manipulacije

newImagesFolder = "resources\\new_images\\"
finalSize = (256, 256)



def resize(image):
    image = image.resize(finalSize)
    image = PIL.ImageOps.exif_transpose(image)
    return image


def mirror(image):
    image = PIL.ImageOps.exif_transpose(image)
    image = PIL.ImageOps.mirror(image)
    
    return image

def croppedCorners(image):
    w, h = image.size

    topleft  = (0, 0, 3*w/4, 3*h/4)
    topright = (w/4, 0, 3*w/4, h)
    botleft  = (0, h/4, 3*w/4, h)
    botright = (w/4, h/4, w, h)


    croppedImages = []
    croppedImages.append(image.crop(topleft))
    #croppedImages.append(image.crop(topright))   
    #croppedImages.append(image.crop(botleft))
    croppedImages.append(image.crop(botright))

    return croppedImages



def manipulateImage(image, imageName):
    imageName = imageName[:-4]

    finalImages = []
    
    finalImages.append(resize(image))
    finalImages.append(resize(mirror(image)))

    croppedImages = croppedCorners(image)
    for im in croppedImages:
        finalImages.append(resize(im))

    i = 1
    for im in finalImages:
        im.save(newImagesFolder + imageName + "_" + str(i) + ".jpg")
        i+=1





for imageName in os.listdir(originalImagesFolder):
    try:
        dest = originalImagesFolder + imageName
        with Image.open(dest) as im:
            manipulateImage(im, imageName)
            print(imageName + " done")
            lll = 1

    except OSError:
        pass

print("manipulation done!")


#
#   Pravljenje bas dataset strukturu direktorija
#
#   Folderi training i testing moraju biti PRAZNI prije pokretanja!!
#
#

trainId = 1
testId = 1

parentDirectoryTraining = "resources\data_set\\training"
parentDirectoryTesting = "resources\data_set\\testing"

for imageName in os.listdir(newImagesFolder):
    dest = newImagesFolder + imageName
    im = Image.open(newImagesFolder + imageName)
    

    # training set
    if(random.random() > 0.2):
        os.mkdir(os.path.join(parentDirectoryTraining, str(trainId)))
        im.save(parentDirectoryTraining + "\\" + str(trainId) + "\\color.jpg")
        bw = im.convert("L")
        bw.save(parentDirectoryTraining + "\\" + str(trainId) + "\\bw.jpg")
        trainId += 1

    # testing set
    else:
        os.mkdir(os.path.join(parentDirectoryTesting, str(testId)))
        im.save(parentDirectoryTesting + "\\" + str(testId) + "\\color.jpg")
        bw = im.convert("L")
        bw.save(parentDirectoryTesting + "\\" + str(testId) + "\\bw.jpg")
        testId += 1



