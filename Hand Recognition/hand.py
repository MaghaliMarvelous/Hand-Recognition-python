import cv2
import mediapipe as mp
import pyautogui
import threading
import time

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,  # Maksimal mendeteksi 1 tangan
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5
)

# Variabel Global
frame = None  # Untuk menyimpan frame video yang diambil
last_action_time = time.time()  # Waktu terakhir aksi dilakukan
gesture_text = ""  # Teks gestur yang akan ditampilkan
frame_skip = 2  # Interval untuk melewati frame agar lebih efisien
frame_count = 0  # Penghitung frame

# Mulai menangkap video dari kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def capture_frames():
    """ Fungsi untuk menangkap frame video secara terus-menerus dalam thread terpisah """
    global frame
    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            break
        frame = new_frame  # Simpan frame terbaru ke variabel global
        time.sleep(0.01)  # Tambahkan delay kecil untuk mengurangi beban CPU

# Jalankan fungsi pengambilan gambar di thread terpisah
cap_thread = threading.Thread(target=capture_frames, daemon=True)
cap_thread.start()

# Loop utama untuk memproses video
while cap.isOpened():
    if frame is None:
        continue  # Tunggu hingga frame pertama tersedia

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Melewati frame tertentu untuk efisiensi

    # Salin frame dan lakukan flipping agar sesuai tampilan pengguna
    processed_frame = cv2.flip(frame.copy(), 1)
    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    # Proses deteksi tangan
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar titik-titik tangan pada frame
            mp_drawing.draw_landmarks(processed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ambil koordinat landmark untuk mendeteksi gestur
            landmarks = hand_landmarks.landmark
            fingers = [8, 12, 16, 20]  # Ujung jari telunjuk, tengah, manis, kelingking
            thumb_tip = landmarks[4]   # Ujung ibu jari
            thumb_base = landmarks[2]  # Sendi ibu jari

            # Konversi koordinat normalisasi ke piksel
            h, w, _ = processed_frame.shape
            index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            middle_x, middle_y = int(landmarks[12].x * w), int(landmarks[12].y * h)

            # Cek apakah semua jari terbuka (telapak tangan terbuka)
            fingers_up = all(landmarks[f].y < landmarks[f - 2].y for f in fingers)
            thumb_up = thumb_tip.y < thumb_base.y  

            # Deteksi gestur dengan cooldown 0.3 detik agar tidak spam
            if time.time() - last_action_time > 0.3:  
                if fingers_up and thumb_up:
                    pyautogui.press("up")
                    gesture_text = "TELAPAK TERBUKA - ATAS"
                elif index_y < middle_y:
                    pyautogui.press("up")
                    gesture_text = "ATAS"
                elif index_y > thumb_y:
                    pyautogui.press("down")
                    gesture_text = "BAWAH"
                elif index_x < thumb_x:
                    pyautogui.press("left")
                    gesture_text = "KIRI"
                elif index_x > thumb_x:
                    pyautogui.press("right")
                    gesture_text = "KANAN"

                last_action_time = time.time()  # Perbarui waktu aksi terakhir

    # Tampilkan teks gestur pada layar
    if gesture_text:
        cv2.putText(processed_frame, f"{gesture_text}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

    # Tampilkan video dengan hasil deteksi tangan
    cv2.imshow("Hand Gesture Recognition", processed_frame)

    # Keluar dari program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan resource setelah program selesai
cap.release()
cv2.destroyAllWindows()
