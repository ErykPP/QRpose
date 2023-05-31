import cv2
import numpy as np
from pyzbar import pyzbar

def euler_angles_to_rotation_vector(roll, pitch, yaw):
    rmat, _ = cv2.Rodrigues(np.array([roll, pitch, yaw]))
    rvec, _ = cv2.Rodrigues(rmat)
    return rvec

def draw_axis(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length=0.5):
    # Wymiary osi
    axis_points_3d = np.float32([[0, 0, 0],  # Początek osi
                                 [axis_length, 0, 0],  # Os X
                                 [0, axis_length, 0],  # Os Y
                                 [0, 0, -axis_length]])  # Os Z

    # Projekcja punktów osi na obraz
    axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs)

    # Konwersja punktów na odpowiedni format
    axis_points_2d = np.int32(axis_points_2d).reshape(-1, 2)

    # Rysowanie osi układu współrzędnych
    image = cv2.line(image, tuple(axis_points_2d[0]), tuple(axis_points_2d[1]), (255, 0, 0), 3)  # Os X - niebieski
    image = cv2.line(image, tuple(axis_points_2d[0]), tuple(axis_points_2d[2]), (0, 255, 0), 3)  # Os Y - zielony
    image = cv2.line(image, tuple(axis_points_2d[0]), tuple(axis_points_2d[3]), (0, 0, 255), 3)  # Os Z - czerwony

    return image

def estimate_pose(qr_code, camera_matrix, dist_coeffs):
    # Wymiary kodu QR
    qr_size = 2.8  # Domyślny rozmiar kodu QR (w cm)

    # Wierzchołki kodu QR
    qr_points_3d = np.float32([[-qr_size / 2, -qr_size / 2, 0],
                               [qr_size / 2, -qr_size / 2, 0],
                               [qr_size / 2, qr_size / 2, 0],
                               [-qr_size / 2, qr_size / 2, 0]])

    # Wierzchołki obrazu kodu QR
    qr_points_2d = np.float32([qr_code[0], qr_code[1], qr_code[2], qr_code[3]])

    # Estymacja pozycji kodu QR
    _, rvec, tvec = cv2.solvePnP(qr_points_3d, qr_points_2d, camera_matrix, dist_coeffs)

    # Przekształcenie wektora rotacji na macierz rotacji
    rmat, _ = cv2.Rodrigues(rvec)

    # Konwersja macierzy rotacji na kąty roll, pitch, yaw
    roll, pitch, yaw = rotation_matrix_to_euler_angles(rmat)

    return tvec, roll, pitch, yaw


def rotation_matrix_to_euler_angles(rmat):
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0

    return x, y, z


def main():
    # Wczytaj kalibrację kamery
    camera_matrix = np.load("calibration_matrix.npy")
    dist_coeffs = np.load("distortion_coefficients.npy")

    # Utwórz obiekt kamery
    cap = cv2.VideoCapture(0)

    while True:
        # Wczytaj obraz z kamery
        ret, frame = cap.read()

        if not ret:
            break

        # Skonwertuj obraz na odcienie szarości
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Wykryj kody QR na obrazie
        barcodes = pyzbar.decode(gray)

        # Przetwórz wykryte kody QR
        for barcode in barcodes:
            # Wyodrębnij współrzędne wierzchołków kodu QR
            barcode_points = barcode.polygon

            # Konwertuj punkty na format obsługiwany przez solvePnP
            qr_code_points = np.array([(p.x, p.y) for p in barcode_points], dtype=np.float32)

            # Estymuj pozycję kodu QR
            tvec, roll, pitch, yaw = estimate_pose(qr_code_points, camera_matrix, dist_coeffs)
            rvec = euler_angles_to_rotation_vector(roll, pitch, yaw)
            frame = draw_axis(frame, camera_matrix, dist_coeffs, rvec, tvec)

            # Wyświetl wyniki
            barcode_data = barcode.data.decode("utf-8")
            print("-------------------")
            print(barcode_data)
            print("Translation vector (x, y, z):", tvec)
            print("Rotation angles (roll, pitch, yaw):", roll, pitch, yaw)
            print()

            # Narysuj prostokąt wokół kodu QR
            cv2.polylines(frame, [np.int32(qr_code_points)], True, (0, 255, 0), 2)

            # Wyświetl odczytaną zawartość kodu QR

            cv2.putText(frame, barcode_data, (barcode_points[0].x, barcode_points[0].y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Wyświetl wynikowy obraz z zaznaczonymi kodami QR
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()