function [bboxes, detectedImg, scores] = feature_detect(gray)
    name = 'tiny-yolov3-coco';
    detector = yolov3ObjectDetector(name);
    analyzeNetwork(detector.Network);
    img = preprocess(detector,gray);
    img = im2single(img);
    [bboxes,scores,labels] = detect(detector,img,'DetectionPreprocessing','none');
    detectedImg = insertObjectAnnotation(img,'Rectangle',bboxes,labels);

end