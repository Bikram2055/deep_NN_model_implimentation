from django.shortcuts import render
from .models import Images
from tensorflow import keras
import cv2 as cv
import numpy as np


class ImageProcessing:
    def __init__(self):
        self.img_width = 224
        self.img_height = 224
        self.sigmaX = 6
        self.tol = 7

    def cropping_2D(self, img, is_cropping=False):
        # cropping in grayscale images
        mask = img > self.tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    def cropping_3D(self, img, is_cropping=False):
        # cropping in RGB image
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        mask = gray_img > self.tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if check_shape == 0:  # if too dark return to original image
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]  # first channel
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]  # second channel
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]  # third channel
            return img

    def Gaussian_blur(self, img, is_Blur=False):
        # Apply gaussian filter in image for smoothing
        # original image and blurred image is blended to create new image
        img = cv.addWeighted(img, 4, cv.GaussianBlur(img, (0, 0), self.sigmaX), -4, 128)
        return img

    def draw_circle(self, img, is_circle=True):
        # draw a circle of specified radius from centre
        center_x = int(self.img_width / 2)
        center_y = int(self.img_height / 2)
        radius = np.amin((center_x, center_y))
        circle_img = np.zeros((self.img_height, self.img_width), np.uint8)
        cv.circle(circle_img, (center_x, center_y), int(radius), 1, -1)
        img = cv.bitwise_and(img, img, mask=circle_img)
        return img

    def img_preprocessing(self, img, is_cropping=True, is_Blur=True):
        if img.ndim == 2:
            img = self.cropping_2D(img, is_cropping)
        else:
            img = self.cropping_3D(img, is_cropping)

        img = cv.resize(img, (self.img_width, self.img_height))
        img = self.draw_circle(img)
        img = self.Gaussian_blur(img, is_Blur)
        return img


def prediction(path):
    img = cv.imread(path)
    img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgObj1 = ImageProcessing()  # Object creation
    img1 = imgObj1.img_preprocessing(img1)
    img1 = imgObj1.cropping_3D(img1)
    img1 = cv.resize(img1, (224, 224))
    img1 = imgObj1.draw_circle(img1)
    img1 = imgObj1.Gaussian_blur(img1)
    model = keras.models.load_model('vgg_model.h5')
    result = np.argmax(model.predict(img1.reshape(1, 224, 224, 3)))
    return result


# Create your views here.
def index(request):
    if request.method == "POST":
        image = request.FILES['image']
        data = Images.objects.create(image=image)
        Images.save(data)
    else:
        pass
    image = Images.objects.all().last()
    result = prediction(f'media/{image.image}')
    results = ['NO DR', 'MILD', 'MODERATE', 'SEVERE', 'PROLIFERATIVE']
    result = results[result]
    return render(request, 'index.html', {'image': image, 'result': result})
