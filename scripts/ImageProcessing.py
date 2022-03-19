from PIL import Image, ImageEnhance
import os


def processing(_filename: str):
    im = Image.open(_filename)

    _filename = _filename[0: len(_filename) - 4]

    enhancer = ImageEnhance.Brightness(im)

    for i in [90, 180, 270]:
        try:
            im_output = im.rotate(i)
            im_output.save(_filename + "_" + str(i) + ".jpg")
        except Exception:
            print(_filename + " ignored")

    factor = 0.5
    try:
        im_output = enhancer.enhance(factor)
        for i in [0, 90, 180, 270]:
            im_output = im_output.rotate(i)
            im_output.save(_filename + "_dark_" + str(i) + ".jpg")
    except Exception:
        print(_filename + " ignored")

    factor = 1.5
    try:
        im_output = enhancer.enhance(factor)
        for i in [0, 90, 180, 270]:
            im_output = im_output.rotate(i)
            im_output.save(_filename + "_bright_" + str(i) + ".jpg")
    except Exception:
        print(_filename + " ignored")


for rep in os.listdir('../images'):
    for fruit in os.listdir('../images/' + rep):
        for file in os.listdir('../images/' + rep + "/" + fruit):
            processing('../images/' + rep + "/" + fruit + "/" + file)
