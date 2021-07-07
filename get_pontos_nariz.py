import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates, DrawingSpec, RED_COLOR

from utils import getPointsFaceMesh2

cap = cv2.VideoCapture(1)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

PRESENCE_THRESHOLD = 0.5
VISIBILITY_THRESHOLD = 0.5
RGB_CHANNELS = 3
landmark_drawing_spec = DrawingSpec(color=RED_COLOR)





# pontos_selecionados = [3, 4, 5, 6, 8, 9, 44, 45, 48, 49, 51, 59, 60, 64, 75, 79, 94, 97, 98, 99, 115, 125, 129,
# 131, 134, 141, 166, 168, 174, 195, 196, 197, 198, 203, 206, 207, 209, 217, 218, 219, 220, 236, 237, 238, 239, 240,
# 241, 242, 248, 250, 274, 275, 278, 279, 281, 289, 290, 294, 305, 309, 326, 327, 328, 331, 344, 354, 355, 358, 360,
# 363, 370, 392, 399, 419, 420, 423, 426, 427, 429, 437, 438, 439, 440, 455, 456, 457, 458, 459, 460, 461, 462]

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
current_index = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    pxs = getPointsFaceMesh2(frame)

    # Our operations on the frame come here
    # for pt in pxs:
    if len(pxs) > 0:
        for i in pontos_selecionados:
            # cv2.putText(frame, str(i), pxs[i][0], cv2.FONT_HERSHEY_SIMPLEX,
            #             0.25, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, pxs[i][0], landmark_drawing_spec.circle_radius,
                       (255, 0, 0) if i == current_index else (0, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    op = cv2.waitKey(1) & 0xFF
    if op == ord('q'):
        break
    # elif op == ord('+'):
    #     pontos_selecionados.append(current_index)
    elif op == ord('s') and current_index + 1 < len(pontos_selecionados):
        current_index = current_index + 1
    elif op == ord('a') and current_index - 1 >= 0:
        current_index = current_index - 1
    elif op == ord(' '):
        print(current_index)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
