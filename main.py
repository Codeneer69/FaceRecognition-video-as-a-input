import os 
import cv2
import face_recognition



video_file = cv2.VideoCapture(os.path.abspath("videos/infinity_war_trailer.mp4"))


length = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))


image_black_panther_1 = face_recognition.load_image_file(os.path.abspath("images/black_panther_1.jpeg"))
image_black_widow_1 = face_recognition.load_image_file(os.path.abspath("images/black_widow.png"))
# image_bruce_banner_1 = face_recognition.load_image_file(os.path.abspath("images/bruce_banner.png"))
image_Captain_America_1 = face_recognition.load_image_file(os.path.abspath("images/Captain_America.png"))
image_loki_1 = face_recognition.load_image_file(os.path.abspath("images/loki.png"))
image_peter_parker_1 = face_recognition.load_image_file(os.path.abspath("images/peter_parker.png"))
image_strange_1 = face_recognition.load_image_file(os.path.abspath("images/strange.png"))
image_thanos_1 = face_recognition.load_image_file(os.path.abspath("images/thanos.png"))
image_tony_stark_1 = face_recognition.load_image_file(os.path.abspath("images/tony_stark.png"))
# image_vision_1 = face_recognition.load_image_file(os.path.abspath("images/vision.png"))
image_wanda_1 = face_recognition.load_image_file(os.path.abspath("images/wanda.png"))

# Generate the face encoding for the image that has been passed.
black_panther_face_1 = face_recognition.face_encodings(image_black_panther_1)[0]
black_widow_face_1 = face_recognition.face_encodings(image_black_widow_1)[0]
# bruce_banner_face_1 = face_recognition.face_encodings(image_bruce_banner_1)[0]
Captain_America_face_1 = face_recognition.face_encodings(image_Captain_America_1)[0]
loki_face_1 = face_recognition.face_encodings(image_loki_1)[0]
peter_parker_face_1 = face_recognition.face_encodings(image_peter_parker_1)[0]
strange_face_1 = face_recognition.face_encodings(image_strange_1)[0]
thanos_face_1 = face_recognition.face_encodings(image_thanos_1)[0]
tony_stark_face_1 = face_recognition.face_encodings(image_tony_stark_1)[0]
# vision_face_1 = face_recognition.face_encodings(image_vision_1)[0]
wanda_face_1 = face_recognition.face_encodings(image_wanda_1)[0]

known_faces = [
    black_panther_face_1,
    black_widow_face_1,
    # bruce_banner_face_1,
    Captain_America_face_1,
    loki_face_1,
    peter_parker_face_1,
    strange_face_1,
    thanos_face_1,
    tony_stark_face_1,
    wanda_face_1
]

facial_points = []
face_encodings = []
facial_number = 0

while True:
    return_value, frame = video_file.read()
    facial_number = facial_number + 1
    
    if not return_value:
        break

    rgb_frame = frame[:, :, ::-1]

    facial_points = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, facial_points)

    facial_names = []

    for encoding in face_encodings:
        match = face_recognition.compare_faces(known_faces, encoding, tolerance=0.60)
        # match = [False, True, True, False , False]

        name = ""
        if match[0]:
            name = "Black Panther"

        if match[1]:
            name = "Black Widow"

        if match[2]:
            name = "Captain"

        if match[3]:
            name = "loki"

        if match[4]:
            name = "spider_man"

        if match[5]:
            name = "strange"

        if match[6]:
            name = "thanos"

        if match[7]:
            name = "Iron man"

        if match[8]:
            name = "wanda"

        facial_names.append(name)

    for (top, right, bottom, left), name in zip(facial_points, facial_names):
        # Enclose the face with the box - Red color 
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Name the characters in the Box created above
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    codec = int(video_file.get(cv2.CAP_PROP_FOURCC))
    fps = int(video_file.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_movie = cv2.VideoWriter("output_{}.mp4".format(facial_number), codec, fps, (frame_width,frame_height))
    print("Writing frame {} / {}".format(facial_number, length))
    output_movie.write(frame)

video_file.release()
output_movie.release()
cv2.destroyAllWindows()