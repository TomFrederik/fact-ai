"""
Adapted from https://github.com/dchen236/FairFace
Citation: Kärkkäinen, K., & Joo, J. (2019). FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age. arXiv preprint arXiv:1908.04913.
Clone date: 14. Jan 2021
License: Creative Commons BY 4.0
"""


from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
import dlib
import os
import argparse


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def detect_face(args, image_filenames, default_max_size=800,size = 300, padding = 0.25):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height
    for index, image_filename in enumerate(image_filenames):
        image_path = os.path.join(args.imgs_load_path, image_filename)
        if index % 1000 == 0:
            print('---%d/%d---' %(index, len(image_filenames)))
        img = dlib.load_rgb_image(image_path)

        old_height, old_width, _ = img.shape

        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
        img = dlib.resize_image(img, rows=new_height, cols=new_width)

        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(image_path))
            continue
        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))
        images = dlib.get_face_chips(img, faces, size=size, padding = padding)
        for idx, image in enumerate(images):
            img_name = image_path.split("/")[-1]
            path_sp = img_name.split(".")
            face_name = os.path.join(args.imgs_save_path,  path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1])
            dlib.save_image(image, face_name)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    # Please create a csv with one column 'img_path', contains the full paths of all images to be analyzed.
    # Also please change working directory to this file.
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_load_path', dest='imgs_load_path', action='store', default='./data/fairface/original/val/',
                        help='path where the input images are loaded from')
    parser.add_argument('--imgs_save_path', dest='imgs_save_path', action='store', default='./data/fairface/cropped/val/',
                        help='path where the cropped images are saved to')
    dlib.DLIB_USE_CUDA = True
    print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)
    args = parser.parse_args()
    ensure_dir(args.imgs_save_path)
    imgs = [img for img in os.listdir(args.imgs_load_path) if os.path.isfile(os.path.join(args.imgs_load_path, img))]
    detect_face(args, imgs)
    print("detected faces are saved at ", args.imgs_save_path)
