from difflib import SequenceMatcher

import cv2
# import easyocr
from deep_translator import GoogleTranslator
from moviepy.editor import *
import paddleocr


def compare_text(text_old, score_sum_old, text_new, score_sum_new, threshold=0.8):

    if not text_old: return True, text_new, score_sum_new

    similarity_ratio = SequenceMatcher(None, text_new, text_old).ratio()

    if similarity_ratio > threshold: # possibly the same text
        if (score_sum_new / len(text_new)) > (score_sum_old / len(text_old)):
            return True, text_new, score_sum_new
        else:
            return False, text_old, score_sum_old
    else: # possibly not the same text
        return True, text_new, score_sum_new

# Set the path to the video file
video_path = './sample2.mp4'

# Load the video
cap = cv2.VideoCapture(video_path)

# Create an OCR reader
# reader = easyocr.Reader(['ja'], gpu=True, model_storage_directory="./")
ocr = paddleocr.PaddleOCR(lang = "japan",
                          use_gpu=True,
                          use_angle_cls=False,
                          show_log = False)

#use_angle_cls to True when using flipped 180 text


translator = GoogleTranslator(source='auto', target='en')

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(height)
print(width)

start_row = int(height * 0.85)
end_row = int(height * 0.95)

start_col = int(width * 0.10)
end_col = int(width * 0.90)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)
print("source framerate:" + str(frame_rate))

frames = []
frame_count = 0
translated = None
# Iterate over each frame in the video

old_text = None
old_score_sum = None
old_position = None
old_translated = None

while (cap.isOpened()):
    # Read the current frame
    ret, frame = cap.read()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_count % 5 == 0:

            # inverse and split
            # region_frame = cv2.bitwise_not(frame[start_row:end_row, start_col:end_col])

            region_frame = frame[start_row:end_row]

            # use bitwise_not to inverse white text - bad for ocr ?

            # cv2.imshow("process", region_frame)
            # cv2.waitKey(1)

            # Perform OCR on the
            # result = reader.readtext(region_frame,decoder = 'wordbeamsearch',
            # width_ths=0.9, paragraph=True)

            result = ocr.ocr(region_frame, cls=False)
            position = None
            texts = ""
            score_sum = 0
            if result[0] is not None: # Not empty list result
                for detections in result:
                    for detection in detections:
                        # boxes = detection[0]
                        # print(detection[1][0])
                        # print(detection[0])
                        if position is None:
                            position = detection[0][0]
                            #position[0] += start_col
                            position[1] += start_row
                        texts += detection[1][0]
                        score_sum += detection[1][1] * len(texts)

                        # img = paddleocr.draw_ocr(region_frame, boxes, text, scores)
                        # Adjust the x,y position of the text to account for the slicing
                        # position[0] -= start_col
                        # position[1] += start_row

                        # cv2.imshow("process", img)
                        # cv2.waitKey(1)

            if texts:

                accepted_result = compare_text(old_text, old_score_sum, texts, score_sum)

                if accepted_result[0]:
                    old_text = texts
                    old_score_sum = score_sum
                    old_position = position
                    old_translated = translator.translate(old_text)
                    #don't call translate too much, resource intensive

                print(old_text)
                frame = cv2.putText(frame,
                                    old_translated,
                                    (int(old_position[0]), int(old_position[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0),
                                    2,
                                    cv2.LINE_4)

            else:
                old_text = None
                old_score_sum = None
                old_position = None
                old_translated = None

        elif old_text:
            print(old_text)
            frame = cv2.putText(frame,
                                old_translated,
                                (int(old_position[0]), int(old_position[1])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2,
                                cv2.LINE_4)

        frames.append(frame)

        print("Progress : " + str(frame_count) + "/" + str(length))
        frame_count += 1

    else:
        break;

cap.release()
video = VideoFileClip(video_path)
audio = video.audio

new_clip = ImageSequenceClip(frames, fps=frame_rate)
clip_length = new_clip.duration

new_clip = new_clip.set_audio(audio)
new_clip.write_videofile("-translated.".join(video_path.rsplit(".", 1)),
                         audio_nbytes=2,
                         temp_audiofile="temp-audio.m4a",
                         remove_temp=True,
                         codec="libx264",
                         audio_codec="aac")







