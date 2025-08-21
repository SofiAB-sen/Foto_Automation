# Character Detection Automation

Este proyecto utiliza modelos YOLOv11 con Ultralytics para realizar reconocimiento automático de numeros de placas vehiculares (ANPR - Automatic Number Plate Recognition) a partir de imágenes. 

---

## ⚙️ Requisitos

### 🔹 Requisitos del sistema

- Python 3.10
- Conda o Miniconda

### 🔹 Crear entorno Conda

```bash
conda env create -f environment.yml
conda activate yolo-detect
```
Sin environment.yml:

```bash
conda create -n yolo-detect python=3.10
conda activate yolo-detect

# Dependencias básicas
conda install -c conda-forge numpy<2 pillow matplotlib tqdm pyyaml pandas seaborn pip requests opencv=4.8.0.76 

#Ultralytics y dependencias adicionales
pip install requests psutil py-cpuinfo "ultralytics-thop>=2.0.0"
```

🚀 Cómo usar

Ejecuta el script Detect.py pasando los argumentos: --process_id --image_path --detected_plate 

```bash
python3 Detect.py --process_id 452CD --image_path /Users/sarboledab/Downloads/plate_2.png --detected_plate CKN3G4
```
