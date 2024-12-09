#################################################################################
# Example of using a custom config directory
#################################################################################

import cv2
import datetime

from pyppbox.standalone import setConfigDir, detectPeople, trackPeople, reidPeople
from pyppbox.utils.visualizetools import visualizePeople


# Use a custom config directory "cfg"
setConfigDir(config_dir="my_config", load_all=True)

input_video = "data/MaxBaLoPNG.mp4" #MaxBaLo_PNG GasbyPNJ
#output_video = "data/output_video2.mp4"
cap = cv2.VideoCapture(input_video)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
#out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    hasFrame, frame = cap.read()

    if hasFrame:

        # Detect people without visualizing
        detected_people, _ = detectPeople(frame, img_is_mat=True, visual=False)

        # Track the detected people
        tracked_people = trackPeople(frame, detected_people, img_is_mat=True)

        # Re-identify the tracked people
        reidentified_people, reid_count = reidPeople(
            frame, 
            tracked_people, 
            img_is_mat=True
        )
        

        # Visualize people in video frame with reid status `show_reid=reid_count`
        visualized_mat = visualizePeople(
            frame, 
            reidentified_people, 
            show_reid=reid_count,
            show_ids = (False,False,False)
        )
        with open("data/logs.txt","a",encoding="utf8") as f:
            for person in reidentified_people:
                f.write(f" {datetime.datetime.now()} : {person.faceid} --> {person.faceid_conf}% & {person.deepid} --> {person.deepid_conf}%\n")
                print(f" {datetime.datetime.now()} : {person.faceid} --> {person.faceid_conf}% & {person.deepid} --> {person.deepid_conf}%\n")
                x, y, w, h = person.box_xywh
                if person.deepid == "Unknown" and person.faceid == "Unknown" :  # Si la personne n'est pas reconnue
                    message = "Hey,I don't know you. Please go to identification"
                elif person.faceid != "Unknown" and person.deepid == "Unknown" :
                    message = f"Hi {person.deepid}, please go and check your new clothes"
                else:
                    message = f"Hi, {person.deepid}"  # Si reconnue, afficher l'ID
                cv2.putText(
                    visualized_mat, message,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2
                )
        cv2.imshow("pyppbox: example_03_custom_cfg_directory.py", visualized_mat)

        #out.write(visualized_mat) #Save the ouput video
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

