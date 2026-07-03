from rfdetr import RFDETRNano
RFDETRNano().export(output_dir='onnx', opset_version=17)
# import cv2
# m = RFDETRNano()
# img = cv2.imread('assets/bus.jpg')
# det = m.predict(img, threshold=0.01)
# print('classes:', det.class_id[:20])
# print('scores :', det.confidence[:20])

