import time

import cognitive_face as CF

KEY = '1f96348454f1477bbfc2ca94de97c329'  # Replace with a valid subscription key (keeping the quotes in place).
CF.Key.set(KEY)

BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)

image_paths = [
    "/Users/mmatak/dev/thesis/datasets/appa-real-release-1/test/005615.jpg_face.jpg"
]

MAX_QUERIES_PER_MINUTE = 20

num_of_queries = 0
for i in range(0, 30):
    for img_url in image_paths:
        faces = CF.face.detect(img_url, face_id=False, landmarks=False, attributes='age')
        age = faces[0]['faceAttributes']['age']
        print(str(i) + ":" + str(age))
        num_of_queries += 1
        if num_of_queries % MAX_QUERIES_PER_MINUTE == 0:
            print("Total queries: " + str(num_of_queries))
            print("going to sleep for a minute ... ")
            start = time.time()
            time.sleep(66)
            end = time.time()
            print("I'm awake! I slept for " + str(end - start) + " seconds.")


def predict(image):
    '''
    Queries MS API for face detection.
    :param image: image path (or URL) or image (as file)
    :return: how old a person in the image is, type: int
    '''
    faces = CF.face.detect(image, face_id=False, landmarks=False, attributes='age')
    age = faces[0]['faceAttributes']['age']
    return int(age)

