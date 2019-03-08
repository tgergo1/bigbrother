import cv2
import face_recognition
import os

cap = cv2.VideoCapture(0)

new_counter = 0

process_this_frame=0
scale = 3
inputs = []

face_locations = []
known_face_encodings = []
known_face_names = []
process_this_frame = 0

locations = []
face_names = []

flipped = []

def learn():
	for filename in os.listdir('.'):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			image = face_recognition.load_image_file(filename)

			encodings=face_recognition.face_encodings(image)
			if len(encodings):
				encoding = face_recognition.face_encodings(image)[0]
				known_face_encodings.append(encoding)
			known_face_names.append(filename[:-4])
		else:
			continue


def drawRect(top,bottom,left,right):
	cv2.rectangle(flipped, (left, top), (right, bottom), (0, 0, 255), 2)
	cv2.rectangle(flipped, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	font = cv2.FONT_HERSHEY_DUPLEX
	cv2.putText(flipped, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

def cropScale(locations,draw):
	crop_img=[]
	for (top, right, bottom, left) in locations:
		top *= scale
		right *= scale
		bottom *= scale
		left *= scale

		crop_img = flipped[top:bottom, left:right]
		if draw is True:
			drawRect(top,bottom,left,right)
	return crop_img

learn()

#cap = cv2.VideoCapture('p.mp4')

if (cap.isOpened()== False): 
	print("Error opening video stream or file")

while True:
	ret, img = cap.read()
	if img is not None:
		#flipped = cv2.flip( img, 1 )
		flipped = img
		small_frame = cv2.resize(flipped, (0, 0), fx=1/scale, fy=1/scale)
		rgb_small_frame = small_frame[:, :, ::-1]
		if process_this_frame == 2:
			process_this_frame=0
			locations = face_recognition.face_locations(rgb_small_frame)
			encodings = face_recognition.face_encodings(rgb_small_frame,locations)
			process_this_frame = not process_this_frame

			face_names = []

			for face_encoding in encodings:
				matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
				name = "Unknown"

				if True in matches:
					first_match_index = matches.index(True)
					name = known_face_names[first_match_index]
				else:
					cv2.imwrite("known"+str(new_counter)+".jpg",cropScale(locations,False))
					new_counter+=1
					learn()


				face_names.append(name)
		else:
			process_this_frame+=1

		cropScale(locations,True)

		cv2.imshow('Video', flipped)
		#cv2.imshow('compressed', rgb_small_frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
