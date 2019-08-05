from PIL import Image
im = Image.open("data/Kang.jpg")
im.rotate(45).show()