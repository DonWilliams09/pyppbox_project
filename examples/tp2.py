import cv2
from pyppbox.utils.visualizetools import visualizePeople
from pyppbox.standalone import MT

def ppb_task(input, main_configs_one, main_configs_two):
    ppbmt_one = MT()  # Multithreading instance for the first reider
    ppbmt_two = MT()  # Multithreading instance for the second reider

    ppbmt_one.setMainModules(main_yaml=main_configs_one)
    ppbmt_two.setMainModules(main_yaml=main_configs_two)

    cap = cv2.VideoCapture(input)
    
    while cap.isOpened():
        hasFrame, frame = cap.read()
        if hasFrame:
            # Process with first reider (Torchreid)
            detected_people_one, _ = ppbmt_one.detectPeople(frame, img_is_mat=True, visual=False)
            tracked_people_one = ppbmt_one.trackPeople(frame, detected_people_one, img_is_mat=True)
            reidentified_people_one, reid_count_one = ppbmt_one.reidPeople(
                frame, tracked_people_one, img_is_mat=True
            )

            # Process with second reider (FaceNet)
            reidentified_people_two, reid_count_two = ppbmt_two.reidPeople(
                frame, tracked_people_one, img_is_mat=True
            )

            # Visualize for first reider
            visualized_frame_one = visualizePeople(
                frame, reidentified_people_one,
                show_ids=(True, True, False),
                show_reid=reid_count_one
            )

            # Ajouter les messages pour le premier reider
            for person in reidentified_people_one:
                x, y, w, h = person.box_xywh
                cid = person.cid
                message = f"Hi, ID {cid}"
                cv2.putText(
                    visualized_frame_one, message,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2
                )

            # Visualize for second reider
            visualized_frame_two = visualizePeople(
                frame, reidentified_people_two,
                show_box=False,
                show_skl=(False, False, 5),
                show_reid=(0, 0),
                show_repspoint=False,
                img_is_mat=True,
                show_ids=(False, False, True)
            )

            # Ajouter les messages pour le deuxième reider
            for person in reidentified_people_two:
                x, y, w, h = person.box_xywh
                cid = person.cid
                message = f"Hi, ID {cid}"
                cv2.putText(
                    visualized_frame_two, message,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2
                )

            # Combine frames side-by-side
            combined_frame = cv2.hconcat([visualized_frame_one, visualized_frame_two])

            scale_percent = 100
            if scale_percent != 100:
                width = int(combined_frame.shape[1] * scale_percent / 100)
                height = int(combined_frame.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized = cv2.resize(combined_frame, dim, interpolation=cv2.INTER_AREA)
                cv2.imshow("pyppbox: Two Reiders on One Video", resized)
            else:
                cv2.imshow("pyppbox: Two Reiders on One Video", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()

    #cv2.destroyAllWindows()

if __name__ == '__main__':
    input_video = "data/gta.mp4"
    main_configs_one = {'detector': 'YOLO_Classic',
                        'tracker': 'SORT',
                        'reider': 'Torchreid'}
    main_configs_two = {'detector': 'YOLO_Classic',
                        'tracker': 'SORT',
                        'reider': 'FaceNet'}

    ppb_task(input_video, main_configs_one, main_configs_two)