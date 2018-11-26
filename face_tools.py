# Copyright (c) 2018 Florent Revest
#               2018 Alyssa Quek
# File distributed under the terms of the MIT license. (See COPYING.MIT)

import os
import cv2
import dlib
import pickle
import numpy as np
import tensorflow as tf
import scipy.spatial as spatial

FACE_LANDMARKS_PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'

## Faces and landmarks detector
def find_faces(img):
    # Detect faces bounding boxes using dlib
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 1)

    ret = []

    for bbox in faces:
        # Detect facial landmarks of each face
        predictor = dlib.shape_predictor(FACE_LANDMARKS_PREDICTOR_PATH)
        shape = predictor(img, bbox)
        points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        im_w, im_h = img.shape[:2]
        left, top = np.min(points, 0)
        right, bottom = np.max(points, 0)
       
        r = 3
        x, y = max(0, left-r), max(0, top-r)
        w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y

        # Store the landmarks in a tuple for the usage of overwrite_face
        ret.append((points - np.asarray([[x, y]]), (x, y, w, h), img[y:y+h, x:x+w]))

    return ret

# Path to the frozen GAN model. This is used to generate new faces
PROGRESSIVE_GAN_MODEL_PATH = 'models/karras2018iclr-celebahq-1024x1024.pkl'

## Fake celebrities faces generation
def dream_faces(number_of_faces):
    # Make TensorFlow less verbose
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # Open a TensorFlow default session
    tf.InteractiveSession()

    # Import the pretrained NVIDIA model
    with open(PROGRESSIVE_GAN_MODEL_PATH, 'rb') as model:
        G, D, Gs = pickle.load(model)

    # Generate random latent vectors
    latents = np.random.randn(number_of_faces, *Gs.input_shapes[0][1:])
    # Generate dummy labels (not used by the network)
    labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

    # Run the model to produce a set of images
    images = Gs.run(latents, labels)

    # Convert images to OpenCV-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
    
    return images

## 3D Transform
def bilinear_interpolate(img, coords):
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T

def grid_coordinates(points):
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1
    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)

def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None

def triangular_affine_matrices(vertices, src_points, dst_points):
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dst_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat

def warp_image(src_img, src_points, dst_points, dst_shape, dtype=np.uint8):
    rows, cols = dst_shape[:2]
    result_img = np.zeros((rows, cols, 3), dtype=dtype)

    delaunay = spatial.Delaunay(dst_points)
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dst_points)))

    process_warp(src_img, result_img, tri_affines, dst_points, delaunay)

    return result_img

## Color Correction
def correct_colours(im1, im2, landmarks1):
    COLOUR_CORRECT_BLUR_FRAC = 0.75
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))

    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur = im2_blur.astype(int)
    im2_blur += 128*(im2_blur <= 1)

    result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

## Copy-and-paste
def apply_mask(img, mask):
    masked_img = np.copy(img)
    num_channels = 3
    for c in range(num_channels):
        masked_img[..., c] = img[..., c] * (mask / 255)

    return masked_img

# Face swapping
def overwrite_face(overwritting_face, overwritten_face, image):
    (src_points, src_shape, src_face) = overwritting_face
    (dst_points, dst_shape, dst_face) = overwritten_face

    w, h = dst_face.shape[:2]

    # Warp Image
    warped_src_face = warp_image(src_face, src_points[:48], dst_points[:48], (w, h))

    # Mask for blending
    mask = np.zeros((w, h), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(dst_points), 255)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask*mask_src, dtype=np.uint8)

    # Correct color
    warped_src_face = apply_mask(warped_src_face, mask)
    face_masked = apply_mask(dst_face, mask)
    warped_src_face = correct_colours(face_masked, warped_src_face, dst_points)

    # Seamless clone
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

    x, y, w, h = dst_shape
    image[y:y+h, x:x+w] = output
