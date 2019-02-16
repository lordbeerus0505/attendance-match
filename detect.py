import cv2
from mtcnn.mtcnn import MTCNN
class Detector:
    def detection(self):
        detector = MTCNN()
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW) #dont show logging of open cv
        while True: 
            #Capture frame-by-frame
            __, frame = cap.read()
            f1=frame
            #Use MTCNN to detect faces
            result = detector.detect_faces(frame)
            if result != []:
                for person in result:
                    bounding_box = person['box']
                    keypoints = person['keypoints']
                    # print("inside detect")
                    #optionally draw bounding box to show
                    # cv2.rectangle(frame,
                    #               (bounding_box[0], bounding_box[1]),
                    #               (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                    #               (0,155,255),
                    #               2)
            
                    # cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
                    # cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
                    # cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
                    # cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
                    # cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
                    #value adjusted as per distance
                    cv2.imwrite('image.jpg',f1[bounding_box[1]-30: bounding_box[1] + bounding_box[3]+30,bounding_box[0]-30:bounding_box[0]+bounding_box[2]+30])
                    return
                    
            #display resulting frame
            # cv2.imshow('frame',frame)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break
        #When everything's done, release capture
        cap.release()
        cv2.destroyAllWindows()
# obj=detector()
# obj.detection()