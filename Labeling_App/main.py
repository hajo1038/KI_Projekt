import PySimpleGUI as sg      
import os
from os.path import exists
import cv2
import base64
import numpy as np
import csv
import pandas as pd

CSV_NAME = "Labels.csv"

def already_labeled(filename):
    if exists("./" + CSV_NAME):
        df = pd.read_csv(CSV_NAME, sep=",")
        return filename in df.iloc[:, 0].values
    else:
        return False

def next_image(image_count, image_list, images_path):
    image_count += 1
    if image_count >= len(image_list):
        window["-Filename-"].update("Alle Bilder wurden gelabeled!")
        window["-Input-"].update(disabled=True)
        return False
    if ".png" not in image_list[image_count]:
        image_count += 1
    window["-Filename-"].update(image_list[image_count])
    im = cv2.imread(images_path + "/" + image_list[image_count])
    print(images_path + "/" + image_list[image_count])
    im = cv2.resize(im, (600, 600))
    image_array = np.array(im)
    _, encoded_image = cv2.imencode(".png", image_array)
    base64_image = base64.b64encode(encoded_image)
    window["-Image-"].update(source=base64_image)
    return True


sg.theme('DarkAmber')    # Keep things interesting for your users

layout = [[sg.Text("Ordner mit den Bildern auswählen:")],
          [sg.FolderBrowse(button_text = "Durchsuchen", initial_folder="/Users/jonathanhaller/Documents/Studium/Master/Verfahren_der_KI/KI_Projekt", enable_events=True, key="-Folder-")],
          [sg.Text("", key="-Filename-")],
          [sg.Button("Löschen", key="-Delete-")],
          [sg.Image(source=None, enable_events=True, size=(600, 600), key="-Image-")],
          [sg.Text('Anzahl der Ritter-Sport-Tafeln:')],
          [sg.Input(enable_events=True, key="-Input-")]]


window = sg.Window('Ritter-Sport Labeling', layout, finalize=True)
window['-Input-'].bind("<Return>", "_Enter")

label_dict = {"file": [], "label": []}
all_labeled = False

while True:                             # The Event Loop
    event, values = window.read() 
    print(event, values)
    if event == "-Folder-":
        images_path = values["-Folder-"]
        image_list = sorted(os.listdir(images_path))
        image_count = 0
        while (already_labeled(image_list[image_count])) or (".png" not in image_list[image_count]):
            image_count += 1
            if image_count >= len(image_list):
                window["-Filename-"].update("Alle Bilder wurden gelabeled!")
                window["-Input-"].update(disabled=True)
                window["-Delete-"].update(disabled=True)
                all_labeled = True
                break
        if all_labeled:
            continue
        im = cv2.imread(images_path + "/" + image_list[image_count])
        print(images_path + "/" + image_list[image_count])
        im = cv2.resize(im, (600,600))
        image_array = np.array(im)
        _, encoded_image = cv2.imencode(".png", image_array)
        base64_image = base64.b64encode(encoded_image)
        window["-Image-"].update(source=base64_image)
        window["-Filename-"].update(image_list[image_count])
    if event == "-Input-" + "_Enter":
        if exists(CSV_NAME):
            label = values["-Input-"]
            filename = image_list[image_count]
            with open(CSV_NAME, "a", newline="") as file:
                writer = csv.writer(file)
                # Write the values row by row
                writer.writerow([filename, label])
        else:
            label = values["-Input-"]
            # save file name from when enter was pressed
            label_dict["file"].append(image_list[image_count])
            label_dict["label"].append(label)
            keys = label_dict.keys()
            # Extract the values from the dictionary
            csv_values = zip(*label_dict.values())
            with open(CSV_NAME, "w", newline="") as file:
                writer = csv.writer(file)
                # Write the column headers
                writer.writerow(keys)
                # Write the values row by row
                writer.writerows(csv_values)
        window["-Input-"].update("")
        # increase image_count to load next image
        image_count += 1
        if image_count >= len(image_list):
            window["-Filename-"].update("Alle Bilder wurden gelabeled!")
            window["-Input-"].update(disabled=True)
            window["-Delete-"].update(disabled=True)
            continue
        while (already_labeled(image_list[image_count])) or (".png" not in image_list[image_count]):
            image_count += 1
            if image_count >= len(image_list):
                window["-Filename-"].update("Alle Bilder wurden gelabeled!")
                window["-Input-"].update(disabled=True)
                window["-Delete-"].update(disabled=True)
                all_labeled = True
                break
        if all_labeled:
            continue
        window["-Filename-"].update(image_list[image_count])
        im = cv2.imread(images_path + "/" + image_list[image_count])
        print(images_path + "/" + image_list[image_count])
        im = cv2.resize(im, (600, 600))
        image_array = np.array(im)
        _, encoded_image = cv2.imencode(".png", image_array)
        base64_image = base64.b64encode(encoded_image)
        window["-Image-"].update(source=base64_image)

    if event == "-Delete-":
        os.remove(images_path + "/" + image_list[image_count])

        image_count += 1
        if image_count >= len(image_list):
            window["-Filename-"].update("Alle Bilder wurden gelabeled!")
            window["-Input-"].update(disabled=True)
            continue
        while ".png" not in image_list[image_count]:
            image_count += 1
        window["-Filename-"].update(image_list[image_count])
        im = cv2.imread(images_path + "/" + image_list[image_count])
        print(images_path + "/" + image_list[image_count])
        im = cv2.resize(im, (600, 600))
        image_array = np.array(im)
        _, encoded_image = cv2.imencode(".png", image_array)
        base64_image = base64.b64encode(encoded_image)
        window["-Image-"].update(source=base64_image)

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

window.close()