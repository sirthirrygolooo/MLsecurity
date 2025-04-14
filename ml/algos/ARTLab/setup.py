import os
print(f'[*] Lancement {os.path.basename(__file__)}')
try:
    import kagglehub
    import shutil
    import pandas as pd
    from PIL import Image
except ImportError:
    try:
        os.system('pip install -r requirements.txt')
    except:
        print('Erreur d\'installation des dépendances')

# dossier img
img_dir = 'img'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
    print(f"[*] Dossier '{img_dir}' créé avec succès.")
else:
    print(f"[*] Le dossier '{img_dir}' existe déjà.")

# Download and move adni_dataset
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

# Download and move lfw_dataset
if not os.path.exists('lfw_dataset'):
    try:
        print("[+] Downloading lfw_dataset...")
        path = kagglehub.dataset_download("your_dataset_owner/lfw_dataset")

        new_path = os.path.join(os.path.dirname(path), 'lfw_dataset')
        os.rename(path, new_path)

        shutil.move(new_path, os.getcwd())

        print("[*] Dataset téléchargé et déplacé avec succès.")
    except Exception as e:
        print(f"[!] Erreur lors du téléchargement ou du déplacement du dataset : {e}")
else:
    print("[*] Le dataset lfw_dataset existe déjà dans le répertoire de travail.")

csv_path = os.path.join('adni_dataset', 'train.csv')

try:
    df = pd.read_csv(csv_path)

    print("################### Train.csv ###################")
    print(df.head())
    print("...\n################################################")

    image_name = df.iloc[0, 0]

    folder_name = image_name.split('-')[0]

    image_path = os.path.join('adni_dataset', 'ADNI_IMAGES', 'png_images', folder_name, image_name)
    image_path += '.png'
    print(image_path)

    image = Image.open(image_path)
    image.show()
except FileNotFoundError:
    print("[!] Le fichier CSV ou l'image spécifiée n'existe pas.")
except Exception as e:
    print(f"[!] Erreur lors de la lecture du fichier CSV ou de l'affichage de l'image : {e}")