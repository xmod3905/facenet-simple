from pustaka import helper, Facenet
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import argparse

myFaceNet = Facenet.loadModel()

def main(args):
    face_detector = helper.get_dlib_face_detector()
    image = Image.open(args.image).convert("RGB")
    i_w, i_h = image.size
    landmarks = face_detector(image)
    draw = ImageDraw.Draw(image)
    for i,landmark in enumerate(landmarks):
        linep = helper.crop_face(image, landmark, expand=.78)
        face = helper.align_and_crop_face(image, linep, 160)
        linep = linep[0]
        face = np.asarray(face)
        face = face.astype("float32")
        mean, std = face.mean(), face.std()
        face = (face-mean)/std
        face = np.expand_dims(face, axis=0)
        signature = myFaceNet.predict(face, verbose=0)
        draw.rectangle([(linep[0],linep[1]),(linep[2],linep[3])], fill=None, outline="green", width=3)
        draw.ellipse([(linep[0],linep[1]),(linep[2],linep[3])], fill=None, outline="blue", width=3)
        print(f"Wajah ke-{(i+1)}", signature)
    image.show()
            

if __name__ == '__main__':
    helper.clear_shel()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--image',
        type=str, 
        default="./test.jpg",
        help="select file path"
    )
    args = parser.parse_args()
    main(args)