import torch
print(torch.cuda.is_available())  # Devrait retourner True si le GPU est disponible
print(torch.cuda.get_device_name(0))  # Devrait imprimer le nom de votre GPU
