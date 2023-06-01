import PySimpleGUI as sg      
import os
import cv2
import base64
import numpy as np
import csv

sg.theme('DarkAmber')    # Keep things interesting for your users

layout = [[sg.Text("Ordner mit den Bildern auswählen:")],
          [sg.FolderBrowse(button_text = "Durchsuchen", initial_folder="/Users/jonathanhaller/Documents/Studium/Master/Verfahren_der_KI/KI_Projekt", enable_events=True, key="-Folder-")],
          [[sg.Text("Name für die csv-Datei", key="-csv_name-")],[sg.Input(key="-csv_input-")]],
          [sg.Text("", key="-Filename-")],
          [sg.Image(source=None, enable_events=True, size=(400, 400), key="-Image-")],
          [sg.Text('Anzahl der Ritter-Sport-Tafeln:')],
          [sg.Input(enable_events=True, key="-Input-")]]


window = sg.Window('Ritter-Sport Labeling', layout, finalize=True)
window['-Input-'].bind("<Return>", "_Enter")

label_dict = {"file": [], "label": []}

while True:                             # The Event Loop
    event, values = window.read() 
    print(event, values)
    if event == "-Folder-":
        images_path = values["-Folder-"]
        image_list = os.listdir(images_path)
        image_count = 0
        im = cv2.imread(images_path + "/" + image_list[0])
        print(images_path + "/" + image_list[0])
        im = cv2.resize(im, (400,400))
        image_array = np.array(im)
        _, encoded_image = cv2.imencode(".png", image_array)
        base64_image = base64.b64encode(encoded_image)
        window["-Image-"].update(source=base64_image)
        window["-Filename-"].update(image_list[0])
    if event == "-Input-" + "_Enter":
        label = values["-Input-"]
        # save file name from when enter was pressed
        label_dict["file"].append(image_list[image_count])
        label_dict["label"].append(label)
        window["-Input-"].update("")
        keys = label_dict.keys()
        # Extract the values from the dictionary
        csv_values = zip(*label_dict.values())
        with open("output.csv", "w", newline="") as file:
            writer = csv.writer(file)
            # Write the column headers
            writer.writerow(keys)
            # Write the values row by row
            writer.writerows(csv_values)
        # increase image_count to load next image
        image_count += 1

        if image_count >= len(image_list):
            window["-Filename-"].update("Alle Bilder wurden gelabeled!")
            window["-Input-"].update(disabled=True)
            continue
        if ".png" not in image_list[image_count]:
            image_count += 1

        window["-Filename-"].update(image_list[image_count])
        im = cv2.imread(images_path + "/" + image_list[image_count])
        print(images_path + "/" + image_list[image_count])
        im = cv2.resize(im, (400, 400))
        image_array = np.array(im)
        _, encoded_image = cv2.imencode(".png", image_array)
        base64_image = base64.b64encode(encoded_image)
        window["-Image-"].update(source=base64_image)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break

window.close()