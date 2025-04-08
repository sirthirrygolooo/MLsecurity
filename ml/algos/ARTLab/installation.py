try :
    import kagglehub
    import shutil
    import os
    import pandas as pd
    from PIL import Image
except ImportError:
    try :
        os.system('pip install -r requirements.txt')
    except :
        print('Erreur d\'installation des dépendances')

if not os.path.exists('adni_dataset'):
    try:
        print("[+] Downloading adni_dataset...")
        path = kagglehub.dataset_download("proutkarshtiwari/adni-images-for-alzheimer-detection")

        new_path = os.path.join(os.path.dirname(path), 'adni_dataset')
        os.rename(path, new_path)

        shutil.move(new_path, os.getcwd())

        print("[*] Dataset téléchargé et déplacé avec succès.")
    except Exception as e:
        print(f"[!] Erreur lors du téléchargement ou du déplacement du dataset : {e}")
else:
    print("[*] Le dataset existe déjà dans le répertoire de travail.")

csv_path = os.path.join('adni_dataset', 'train.csv')

try:
    df = pd.read_csv(csv_path)

    print("###################Train.csv ###################")
    print(df.head())
    print("...\n##############################################")

    image_path = os.path.join('adni_dataset', 'ADNI_IMAGES', df.iloc[0, 0])
    print(image_path)
    #image = Image.open(image_path)
    #image.show()
except FileNotFoundError:
    print("[!] Le fichier CSV ou l'image spécifiée n'existe pas.")
except Exception as e:
    print(f"[!] Erreur lors de la lecture du fichier CSV ou de l'affichage de l'image : {e}")
