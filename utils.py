import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates, DrawingSpec, RED_COLOR
from PIL import Image


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
    '''

    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


def getCantos(maskFace):
    w, h = maskFace.shape[:2]
    cx = int(w / 2)
    cy = int(h / 2)
    pt1x, pt1y = (0, 0)
    pt2x, pt2y = (0, 0)

    while not maskFace[pt1x, cy]:
        pt1x = pt1x + 1

    while not maskFace[cx, pt1y]:
        pt1y = pt1y + 1

    c1 = (pt1x, pt1y)
    pt2x = cx + pt1x
    pt2y = int((cy * 1.3) + pt1y)
    c2 = (pt2x, pt2y)

    return c1, c2


def trataNan(vet, zero_is_allowed=True, negative_is_allowed=True, maskMap=None):
    mask1 = np.isnan(vet)
    vet[mask1] = -1
    if not zero_is_allowed:
        mask2 = vet == 0
        mask = mask1 + mask2
    else:
        mask = mask1
    if not negative_is_allowed:
        maskN = vet < 0
        mask = mask + maskN
    media = vet[~mask].mean()
    vet[mask] = media

    if maskMap is not None:
        media = vet[maskMap].mean()
        vet[~maskMap] = media
    return vet


def normaliza(x, y, z, g=0):
    x = (x + 2) / 4
    y = (y + 2) / 4
    g = g / 255
    if g == 0:
        return x, y, z
    else:
        return x, y, z, g


# devolve as seguintes matrizes: H, x, y e z, todas com 256x332
@DeprecationWarning
def getGrossCharacteristics(res, img):
    maskFace = res['mask']
    maskVet = maskFace[:, :].flatten()
    canto1, canto2 = getCantos(maskFace)

    x = trataNan(res['X'])
    y = trataNan(res['Y'])
    z = trataNan(res['Z'], negative_is_allowed=False)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = gray[:, :].flatten()
    g = trataNan(g, zero_is_allowed=False, maskMap=maskVet)

    x, y, z, g = normaliza(x, y, z, g)

    x = x.reshape(512, 512)
    y = y.reshape(512, 512)
    z = z.reshape(512, 512)
    g = g.reshape(512, 512)

    xyzg = np.dstack((x, y, z, g))

    x, y = canto1
    w = canto2[0] - x
    h = canto2[1] - y

    xyzg = xyzg[y:y + h, x:x + w]

    img = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_RGB2BGR)
    return img, xyzg


def getGrossCharacteristics2(res, img):
    x = trataNan(res['X'])
    y = trataNan(res['Y'])
    z = trataNan(res['Z'], negative_is_allowed=False)

    x, y, z = normaliza(x, y, z)

    x = x.reshape(512, 512)
    y = y.reshape(512, 512)
    z = z.reshape(512, 512)

    xyz = np.dstack((x, y, z))
    return xyz, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def getPointCloud(res, img):
    r = img[:, :, 0].flatten()
    g = img[:, :, 1].flatten()
    b = img[:, :, 2].flatten()

    x = res['X'].flatten()
    y = res['Y'].flatten()
    z = res['Z'].flatten()
    z = trataNan(z, negative_is_allowed=False)
    maskz = z >= 0
    masky = y < y.max() - 0.1

    mask = maskz == masky

    colors = np.stack((r, g, b), axis=1)[mask]

    points = np.stack((x, y, z), axis=1)[mask]

    invalid_inds = np.any(np.isnan(points), axis=1)
    points_valid = points[invalid_inds == False]
    colors_valid = colors[invalid_inds == False]

    return points_valid, colors_valid

def getPointCloud2(res, img, mask_net):
    r = img[:, :, 0].flatten()
    g = img[:, :, 1].flatten()
    b = img[:, :, 2].flatten()

    x = res['X'].flatten()
    y = res['Y'].flatten()
    z = res['Z'].flatten()
    z = trataNan(z, negative_is_allowed=False)
    maskz = z >= 0
    masky = y < y.max() - 0.1

    mask = maskz == masky
    mask = mask == mask_net.flatten()

    colors = np.stack((r, g, b), axis=1)[mask]

    points = np.stack((x, y, z), axis=1)[mask]

    invalid_inds = np.any(np.isnan(points), axis=1)
    points_valid = points[invalid_inds == False]
    colors_valid = colors[invalid_inds == False]

    return points_valid, colors_valid


def mergePointCloud(pts, colors):
    pts1, pts2 = pts
    colors1, colors2 = colors
    pts = np.concatenate((pts1, pts2))
    colors = np.concatenate((colors1, colors2))
    return pts, colors


# ----------------------------------------------------------------------------------------------------------------------
# media pipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

PRESENCE_THRESHOLD = 0.5
VISIBILITY_THRESHOLD = 0.5
RGB_CHANNELS = 3
landmark_drawing_spec = DrawingSpec(color=RED_COLOR)


def draw_landmarks2(
        image: np.ndarray,
        landmark_list: landmark_pb2.NormalizedLandmarkList):
    if not landmark_list:
        return
    if image.shape[2] != RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    pxs = []
    for landmark_px in idx_to_coordinates.values():
        cv2.circle(image, landmark_px, landmark_drawing_spec.circle_radius,
                   landmark_drawing_spec.color, 1)
        pxs.append(landmark_px)
    return pxs, image, landmark_list.landmark


pontos_selecionados = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                       146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                       78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                       78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308] + [246, 161, 160, 159, 158, 157, 173,
                                                                            33, 7, 163, 144, 145, 153, 154, 155, 133,
                                                                            247, 30, 29, 27, 28, 56, 190,
                                                                            130, 25, 110, 24, 23, 22, 26, 112, 243,
                                                                            113, 225, 224, 223, 222, 221, 189,
                                                                            226, 31, 228, 229, 230, 231, 232, 233, 244,
                                                                            143, 111, 117, 118, 119, 120, 121, 128, 245,
                                                                            156, 70, 63, 105, 66, 107, 55, 193,
                                                                            35, 124, 46, 53, 52, 65] + [466, 388, 387,
                                                                                                        386, 385, 384,
                                                                                                        398,
                                                                                                        263, 249, 390,
                                                                                                        373, 374, 380,
                                                                                                        381, 382, 362,
                                                                                                        467, 260, 259,
                                                                                                        257, 258, 286,
                                                                                                        414,
                                                                                                        359, 255, 339,
                                                                                                        254, 253, 252,
                                                                                                        256, 341, 463,
                                                                                                        342, 445, 444,
                                                                                                        443, 442, 441,
                                                                                                        413,
                                                                                                        446, 261, 448,
                                                                                                        449, 450, 451,
                                                                                                        452, 453, 464,
                                                                                                        372, 340, 346,
                                                                                                        347, 348, 349,
                                                                                                        350, 357, 465,
                                                                                                        383, 300, 293,
                                                                                                        334, 296, 336,
                                                                                                        285, 417,
                                                                                                        265, 353, 276,
                                                                                                        283, 282,
                                                                                                        295] + [168] + [
                          3, 4, 5, 6, 8, 9, 44, 45, 48, 49, 51, 59, 60, 64, 75, 79, 94, 97, 98, 99, 115, 125, 129, 131,
                          134, 141, 166, 168, 174, 195, 196, 197, 198, 203, 206, 207, 209, 217, 218, 219, 220,
                          236, 237, 238, 239, 240, 241, 242, 248, 250, 274, 275, 278, 279, 281, 289, 290, 294, 305, 309,
                          326,
                          327, 328, 331, 344, 354, 355, 358, 360, 363, 370, 392, 399, 419, 420, 423, 426, 427, 429, 437,
                          438,
                          439, 440, 455, 456, 457, 458, 459, 460, 461, 462] + [205] + [425]

# @DeprecationWarning
def getPointsFaceMesh(image: np.ndarray):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    pxs = None
    landmarks3D = []
    if results.multi_face_landmarks:
        if results.multi_face_landmarks is not None:
            face_landmarks = results.multi_face_landmarks[0]
            for land in face_landmarks.landmark:
                landmarks3D.append(land)
            pxs2 = draw_landmarks2(
                image=image,
                landmark_list=face_landmarks)

        return pxs2[0], landmarks3D, image
    return None, None, None


def getPointsFaceMesh2(image: np.ndarray):
    image_rows, image_cols, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    landmarks3D = []
    if results.multi_face_landmarks:
        if results.multi_face_landmarks is not None:
            face_landmarks = results.multi_face_landmarks[0]
            for land in face_landmarks.landmark:
                landmarks3D.append(land)
    pxs = []
    if results.multi_face_landmarks:
        if results.multi_face_landmarks is not None:
            landmark_list = results.multi_face_landmarks[0]
            for idx, landmark in enumerate(landmark_list.landmark):
                pt = [_normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows), landmark.z]
                pxs.append(pt)
    pontos_filtrados = []
    for i in range(len(pxs)):
        if i in pontos_selecionados:
            pontos_filtrados.append(pxs[i])
    return pontos_filtrados
