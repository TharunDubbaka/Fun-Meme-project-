import cv2
import mediapipe as mp
import time

def load(path): return cv2.imread(path)

alluhair = load("C:\\Users\\dubba\\OneDrive\\Desktop\\Machine learning projects\\meme_generation\\memes\\alluhair.jpeg")
ntrsmile = load("C:\\Users\\dubba\\OneDrive\\Desktop\\Machine learning projects\\meme_generation\\memes\\ntrsmile.jpeg")
baane = load("C:\\Users\\dubba\\OneDrive\\Desktop\\Machine learning projects\\meme_generation\\memes\\ntrshock.jpeg")
baby = load("C:\\Users\\dubba\\OneDrive\\Desktop\\Machine learning projects\\meme_generation\\memes\\baby.jpeg")
ideamonkey = load("C:\\Users\\dubba\\OneDrive\\Desktop\\Machine learning projects\\meme_generation\\memes\\ideamonkey.jpeg")
thinkmonkey = load("C:\\Users\\dubba\\OneDrive\\Desktop\\Machine learning projects\\meme_generation\\memes\\thinkmonkey.jpeg")
allulady = load("C:\\Users\\dubba\\OneDrive\\Desktop\\Machine learning projects\\meme_generation\\memes\\allulady.jpeg")
meme_happy = load("C:\\Users\\dubba\\OneDrive\\Desktop\\Machine learning projects\\meme_generation\\memes\\speed.jpeg")

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=2)
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
last_trigger = 0
cooldown = 1.5
current_meme = None

def show_meme(img):
    target_width = 450
    h, w = img.shape[:2]
    scale = target_width / w
    resized = cv2.resize(img, (target_width, int(h * scale)))
    cv2.imshow("Meme", resized)

def is_smiling(lm):
    return abs(lm[61].x - lm[291].x) / (abs(lm[13].y - lm[14].y)+1e-6) > 2.2

def eyes_closed(lm):
    return abs(lm[159].y - lm[145].y) < 0.008 and abs(lm[386].y - lm[374].y) < 0.008

def mouth_open(lm):
    return abs(lm[13].y - lm[14].y) > 0.035

def palm_open(hand):
    tips = [8,12,16,20]
    return all(hand.landmark[t].y < hand.landmark[t-2].y for t in tips)

def is_fist(hand):
    tips = [8,12,16,20]
    folded = sum(hand.landmark[t].y > hand.landmark[t-2].y for t in tips)
    return folded >= 3  

def fist_center(hand):
    return hand.landmark[9]  

def fist_near_mouth(hand, face_lm):
    c = fist_center(hand)
    mouth = face_lm[13]
    return abs(c.x - mouth.x) < 0.05 and abs(c.y - mouth.y) < 0.05

def fist_right_of_chin(hand, face_lm):
    chin = face_lm[152]
    c = fist_center(hand)
    return is_fist(hand) and c.x > chin.x + 0.03 and abs(c.y - chin.y) < 0.06

def hand_near_chin_open(hand, face_lm):
    chin = face_lm[152]
    c = fist_center(hand)
    return palm_open(hand) and abs(c.y - chin.y) < 0.05

def two_palms_open(hand_results, face_lm):
    if not hand_results.multi_hand_landmarks or len(hand_results.multi_hand_landmarks) < 2:
        return False
    chin_y = face_lm[152].y
    return sum(palm_open(h) and abs(h.landmark[9].y - chin_y) < 0.05
               for h in hand_results.multi_hand_landmarks) == 2

cv2.namedWindow("Camera")
cv2.namedWindow("Meme")
cv2.moveWindow("Camera", 0, 0)
cv2.moveWindow("Meme", 800, 0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    if face_results.multi_face_landmarks:
        face_lm = face_results.multi_face_landmarks[0].landmark

        if time.time() - last_trigger > cooldown:

            if eyes_closed(face_lm):
                current_meme = meme_happy

            elif mouth_open(face_lm):
                current_meme = baane

            elif is_smiling(face_lm):
                current_meme = ntrsmile
            if hand_results.multi_hand_landmarks:

                if two_palms_open(hand_results, face_lm):
                    current_meme = alluhair

                else:
                    hand = hand_results.multi_hand_landmarks[0]

                    if is_fist(hand) and fist_near_mouth(hand, face_lm):
                        current_meme = baby

                    elif fist_right_of_chin(hand, face_lm):
                        current_meme = allulady

                    elif hand_near_chin_open(hand, face_lm):
                        current_meme = ideamonkey

            last_trigger = time.time()

    if current_meme is not None:
        show_meme(current_meme)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


