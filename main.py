import cv2
import numpy as np

CAMERA_CODE = 2

cap = cv2.VideoCapture(CAMERA_CODE)

widthTarget = 320
heightTarget = 320 
confidenceThreshold = 0.7
nmsThreshold = 0.3 # lower nms will give more accuracy => less number of boxes

classesFiles = ["coco.names", "openimages.names", "voc.names", "9k.names"] # file that hols class names
classNames = []

for cf in classesFiles:
  with open(cf, "rt") as file:
    classNames += file.read().rstrip("\n").split("\n") # extract all class names form the file as array of strings

# Creating network with yoloV3
modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # delcaring to use openCV as the backend
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # set CPU as our target

def findObjects(outputs, image):
  height, width, channel = image.shape
  boundingBox = [] # the list will contain values of [x, y, w, h] => in the end will be a 2D array
  classIds = [] # the list will contain values of class ids
  confidenceValues = [] #the list wil contain the confidence values from each outputs element
  
  # cx(center x), cy(center y), w (width), h(height), confidence(the value which describe that there is on object within the bounding box)
  # those values are a set of element in the outputs array
  for output in outputs: # loop through each output
    for detection in output: # loop through each element of the output's array
      # need to remove the first 5 elements of the array
      scores = detection[5:]
      classId = np.argmax(scores) # get the index of the highest probability value
      confidence = scores[classId] # get that highest probability value and set it as confidence value
      if confidence > confidenceThreshold:
        # if the confidence value is above 50% confident => save it to the confidenceValues list as good detection
        w, h = int(detection[2]*width), int(detection[3]*height)# since elements in detection array are percentage => convert them to pixels by multiply them with the width and height values from our image
        
        # get the corner value point as detection[0] is center x and detection[1] is center y => we want them as pixels (integer)
        x, y = int((detection[0]*width) - (w/2)), int((detection[1]*height)-(h/2)) 
        boundingBox.append([x, y, w, h]) # append [x, y, w, h] to bounding box array
        classIds.append(classId) # append the accepted classId with confidence higher than the given threshold to the classIds array
        confidenceValues.append(float(confidence)) # append the accepted confidence value to the confidenceValues array as array float
        
  # since there are cases where there will be thousands of bounding boxes in the image, there are high probability that they will overlap with each other 
  # as those overlapped boxes are referencing the same object
  # Therefore, we need to filter those boxes to a single boxes => Apply non maximum suppression to eliminate the overlapping boxes by only taking the box with the highest confidence value and ignore the rest 
  indicesToKeep = cv2.dnn.NMSBoxes(boundingBox, confidenceValues, confidenceThreshold, nmsThreshold)
  
  for i in indicesToKeep:
    i = i[0]
    box = boundingBox[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    cv2.rectangle(image, (x, y), (x+w, y+h), (12, 207, 126), 5) # draw image based on those bounding box parameters
    cv2.rectangle(image, (x, y), (x+200, y-50), (12, 207, 126), -5)
    cv2.putText(image, f"{classNames[classIds[i]].upper()} {int(confidenceValues[i]*100)}%", 
                (x+20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 3)
    
while True:
  try:
    success, image = cap.read()
    blob = cv2.dnn.blobFromImage(image, 1/255, (widthTarget, heightTarget), [0,0,0], 1, crop=False)# the neural network only accept blob as the image format
    net.setInput(blob) # feed blob to neural network
    layerNames = net.getLayerNames() # get all names of the network layer names
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()] # base from the network architecture, it is expected to receive 3 output layers => get names of those output layers

    outputs = net.forward(outputNames)
    findObjects(outputs, image)
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except:
    print("Camera Stream is unable to connect\n---> Attempting to reconnect")

cap.release()
cv2.destroyAllWindows()