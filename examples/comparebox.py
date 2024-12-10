import cv2
from pyppbox.utils.visualizetools import visualizePeople
from pyppbox.standalone import MT

def ppb_task(config_dir_one, config_dir_two):
    ppbmt_one = MT()  # Multithreading instance for the first reider
    ppbmt_two = MT()  # Multithreading instance for the second reider

    #ppbmt_one.setMainModules(main_yaml=main_configs_one)
    #ppbmt_two.setMainModules(main_yaml=main_configs_two)
    ppbmt_one.setConfigDir(config_dir_one, load_all=True)
    ppbmt_two.setConfigDir(config_dir_two, load_all=True)
    

    cap = cv2.VideoCapture(0) # 0 for live streming
    
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
            detected_people_two, _ = ppbmt_two.detectPeople(frame, img_is_mat=True, visual=False)
            tracked_people_two = ppbmt_two.trackPeople(frame, detected_people_two, img_is_mat=True)
            reidentified_people_two, reid_count_two = ppbmt_two.reidPeople(
                frame, tracked_people_two, img_is_mat=True
            )
            
            #reidentified_people_two, reid_count_two = ppbmt_two.reidPeople(
            #    frame, tracked_people_one, img_is_mat=True
            #)

            # Visualize for first reider (Torchreid)
            visualized_frame_one = visualizePeople(
                frame.copy(), reidentified_people_one,  # Utiliser une copie pour éviter les conflits
                show_ids=(False, True, False),
                show_reid=reid_count_one
            )

            # Ajouter les messages pour le premier reider (Torchreid)
            for person in reidentified_people_one:
                x1, y1, w1, h1 = person.box_xywh
                if person.deepid == "Unknown" and person.faceid == "Unknown" :  # Si la personne n'est pas reconnue
                    message = "Not recognized, please go to identification"
                elif person.faceid != "Unknown" and person.deepid == "Unknown" :
                    message = f"Hi {person.deepid}, please go and check your new clothes"
                else:
                    message = f"Hi, {person.deepid}"  # Si reconnue, afficher l'ID
                cv2.putText(
                    visualized_frame_one, message,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2
                )

            # Visualize for second reider (FaceNet)
            visualized_frame_two = visualizePeople(
                frame.copy(), reidentified_people_two,  # Utiliser une copie pour éviter les conflits
                # show_box=False,
                #show_skl=(False, False, 5),
                #show_reid=(0, 0),
                #show_repspoint=False,
                #img_is_mat=True,
                show_ids=(False, True, False),
                show_reid=reid_count_two   
            )

            # Ajouter les messages pour le deuxième reider (FaceNet)
            for person in reidentified_people_two:
                x2, y2, w2, h2 = person.box_xywh
                if person.deepid == "Unknown" and person.faceid == "Unknown" :  # Si la personne n'est pas reconnue
                    message = "Not recognized, please go to identification"
                elif person.faceid != "Unknown" and person.deepid == "Unknown" :
                    message = f"Hi {person.deepid}, please go and check your new clothes"
                else:
                    message = f"Hi {person.deepid}"  # Si reconnue, afficher l'ID
                cv2.putText(
                    visualized_frame_two, message,
                    (x2, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2
                )

            # Combine the two frames side by side (Torchreid on the left, FaceNet on the right)
            combined_frame = cv2.hconcat([visualized_frame_one, visualized_frame_two])

            # Optional: Resize the combined frame
            scale_percent = 100
            if scale_percent != 100:
                width = int(combined_frame.shape[1] * scale_percent / 100)
                height = int(combined_frame.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized = cv2.resize(combined_frame, dim, interpolation=cv2.INTER_AREA)
                cv2.imshow("Torchreid (Left) | FaceNet (Right)", resized)
            else:
                cv2.imshow("Torchreid (Left) | FaceNet (Right)", combined_frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()

    #cv2.destroyAllWindows()

if __name__ == '__main__':
    #input_video = "data/GasbyPNJ.mp4"
    #input_video = cv2.VideoCapture(0)
    config_dir_one = "/Users/don_williams09/TP_IA/Cases Studies/pyppbox/examples/my_config"
    config_dir_two = "/Users/don_williams09/TP_IA/Cases Studies/pyppbox/examples/cfg"
    ppb_task(config_dir_one,config_dir_two)
    
    
    