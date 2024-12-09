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
        #frame2 =frame.copy()
        if hasFrame:
            # Process with first reider
            detected_people_one, _ = ppbmt_one.detectPeople(frame, img_is_mat=True, visual=False)
            tracked_people_one = ppbmt_one.trackPeople(frame, detected_people_one, img_is_mat=True)
            reidentified_people_one, reid_count_one = ppbmt_one.reidPeople(
                frame, tracked_people_one, img_is_mat=True
            )

            # Process with second reider
            #detected_people_two, _ = ppbmt_two.detectPeople(frame, img_is_mat=True, visual=False)
            #tracked_people_two = ppbmt_two.trackPeople(frame, detected_people_two, img_is_mat=True)
            reidentified_people_two, reid_count_two = ppbmt_two.reidPeople(
                frame, tracked_people_one, img_is_mat=True
            )

            # Combine visualizations
            visualized_frame_one = visualizePeople(
                frame, reidentified_people_one,
                show_ids=(True, True, False),
                show_reid=reid_count_one
            )
            visualized_frame_two = visualizePeople(
                frame, reidentified_people_two, 
                show_box=False, 
                show_skl=(False, False, 5),
                show_reid=(0, 0), 
                show_repspoint=False, 
                img_is_mat=True,
                show_ids=(False, False, True),
                #show_reid=reid_count_two
            )

            # Combine frames side-by-side (or overlay annotations)
            combined_frame = cv2.hconcat([visualized_frame_one, visualized_frame_two])
            
            scale_percent = 100 # percent of original size
            if(scale_percent != 100):
                width = int(combined_frame.shape[1] * scale_percent / 100)
                height = int(combined_frame.shape[0] * scale_percent / 100)
                dim = (width, height)
            
                # resize image
                resized = cv2.resize(combined_frame, dim, interpolation = cv2.INTER_AREA)
            
                # Display the combined frame
                cv2.imshow("pyppbox: Two Reiders on One Video", resized)
            else:
                cv2.imshow("pyppbox: Two Reiders on One Video", visualized_frame_two)
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