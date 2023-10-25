from time import time

import cv2
import mediapipe as mp

# Initialize the mediapipe face detection class.
mp_face_detection = mp.solutions.face_detection

# Set up the face detection function.
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

# Initialize the mediapipe face mesh class
mp_face_mesh = mp.solutions.face_mesh

# Set up the face landmarks function for videos
face_mesh_videos = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

# Initialize the mediapipe drawing styles class
mp_drawing_styles = mp.solutions.drawing_styles


def detect_face_landmarks(image, face_mesh):
    rgb_image = image[:, :, ::-1]
    results = face_mesh.process(rgb_image)
    output_image = rgb_image.copy()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=output_image, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=output_image, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

    return output_image, results


if __name__ == "__main__":
    camera_video = cv2.VideoCapture(1)
    cv2.namedWindow('Face Landmarks Detection', cv2.WINDOW_NORMAL)
    time1 = 0

    cv2.namedWindow('Face Landmarks Detection', cv2.WINDOW_NORMAL)
    while camera_video.isOpened():
        ok, frame = camera_video.read()
        if not ok:
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        frame, _ = detect_face_landmarks(frame, face_mesh_videos)
        time2 = time()
        if (time2 - time1) > 0:
            frames_per_second = 1.0 / (time2 - time1)
            cv2.putText(
                frame, f'FPS: {int(frames_per_second)}',
                (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3
            )

        time1 = time2
        cv2.imshow('Face Landmarks Detection', frame)

        k = cv2.waitKey(27)
        if k == 27:
            break

    camera_video.release()
    cv2.destroyAllWindows()
